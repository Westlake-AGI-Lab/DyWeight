"""Script for calculating the CLIP Score."""

import os
import sys
import csv
import click
import tqdm
import torch

# Add project root to path for torch_utils import when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_utils import distributed as dist
from training import dataset
import open_clip
from torchvision import transforms
from torch_utils.download_util import check_file_by_key

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate CLIP score.
    python eval/clip_score.py calc --images=path/to/images
    or running on multiple GPUs (e.g., 4):
    torchrun --standalone --nproc_per_node=4 eval/clip_score.py calc --images=path/to/images
    """

#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=250, show_default=True)
@click.option('--model',  'model_name', help='CLIP model name',                                     type=str, default='ViT-g-14', show_default=True)
@click.option('--pretrained',           help='Pretrained weights tag or local .safetensors path',   type=str, default='laion2b_s34b_b88k', show_default=True)
@click.option('--cache_dir',            help='Local cache dir for offline use (optional)',           type=str, default=None, show_default=True)

@torch.no_grad()
def calc(image_path, batch, model_name, pretrained, cache_dir,
         num_expected=None, seed=0,
         num_workers=3, prefetch_factor=2, device=torch.device('cuda')):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)

    # Load COCO captions.
    prompt_path, _ = check_file_by_key('prompts')
    sample_captions = []
    with open(prompt_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sample_captions.append(row['text'])

    # Load CLIP model.
    # Priority:
    #   1. cache_dir points directly to a .safetensors file  → use it as-is (offline)
    #   2. cache_dir is a directory                          → search for a file whose
    #      name contains the sanitized model name (e.g. "ViT-g-14" → "ViT-g-14"),
    #      falling back to the first .safetensors found     (offline)
    #   3. cache_dir is None                                 → download via pretrained tag (online)
    local_weights = None
    if cache_dir is not None:
        if os.path.isfile(cache_dir) and cache_dir.endswith('.safetensors'):
            local_weights = cache_dir
        elif os.path.isdir(cache_dir):
            # Prefer a file whose path contains the model name to avoid picking the
            # wrong checkpoint when multiple models are cached in the same directory.
            model_key = model_name.replace('/', '-')
            best, fallback = None, None
            for root, _, files in os.walk(cache_dir):
                for fname in files:
                    if fname.endswith('.safetensors'):
                        full = os.path.join(root, fname)
                        if model_key.lower() in full.lower():
                            best = full
                            break
                        if fallback is None:
                            fallback = full
                if best:
                    break
            local_weights = best or fallback

    if local_weights is not None:
        dist.print0(f'Loading {model_name} from local weights: {local_weights}')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=local_weights)
    else:
        dist.print0(f'Loading {model_name} (pretrained={pretrained}) from the internet...')
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (batch * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)


    dist.print0(f'Calculating CLIP score for {len(dataset_obj)} images...')
    total_clip_score = torch.zeros(1, dtype=torch.float64, device=device)
    to_pil = transforms.ToPILImage()
    for batch_idx, (images, _) in enumerate(tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0))):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        batch_indices = rank_batches[batch_idx]
        start_idx = int(batch_indices[0].item())
        end_idx = int(batch_indices[-1].item()) + 1
        prompts = sample_captions[start_idx:end_idx]

        images = torch.stack([preprocess(to_pil(img)) for img in images], dim=0).to(device)
        text = tokenizer(prompts).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        sd_clip_score = 100 * (image_features * text_features).sum(axis=-1)
        total_clip_score += sd_clip_score.sum(dtype=torch.float64)

    torch.distributed.all_reduce(total_clip_score)
    avg_clip_score = total_clip_score.item() / len(dataset_obj)
    dist.print0(f'CLIP score: {avg_clip_score}')

    torch.distributed.barrier()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
