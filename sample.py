import os
import re
import csv
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*No device id is provided.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*using GPU.*device used by this process.*')

from torch import autocast
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch_utils.download_util import check_file_by_key
from training.loss import get_solver_fn

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Load pre-trained models from the LDM codebase (https://github.com/CompVis/latent-diffusion)
# and Stable Diffusion codebase (https://github.com/CompVis/stable-diffusion)

def load_ldm_model(config, ckpt, verbose=False):
    from models.ldm.util import instantiate_from_config
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        dist.print0("missing keys:")
        dist.print0(m)
    if len(u) > 0 and verbose:
        dist.print0("unexpected keys:")
        dist.print0(u)
    return model

#----------------------------------------------------------------------------

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None):
    model_path, classifier_path = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')

    if dataset_name in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64']:         # models from EDM
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
        net.sigma_min = 0.002
        net.sigma_max = 80.0
        model_source = 'edm'
    elif dataset_name in ['lsun_bedroom']:                                   # models from Consistency Models
        from models.cm.cm_model_loader import load_cm_model
        from models.networks_edm import CMPrecond
        net = load_cm_model(model_path)
        net = CMPrecond(net).to(device)
        model_source = 'cm'
    else:
        if guidance_type == 'cg':                                            # models from ADM (classifier guidance)
            assert classifier_path is not None
            from models.guided_diffusion.cg_model_loader import load_cg_model
            from models.networks_edm import CGPrecond
            net, classifier = load_cg_model(model_path, classifier_path)
            net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
            model_source = 'adm'
        elif guidance_type in ['uncond', 'cfg']:                             # models from LDM
            from omegaconf import OmegaConf
            from models.networks_edm import CFGPrecond
            if dataset_name in ['lsun_bedroom_ldm']:
                config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            elif dataset_name in ['ms_coco']:
                assert guidance_type == 'cfg'
                config = OmegaConf.load('./models/ldm/configs/stable-diffusion/v1-inference.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=4, guidance_rate=guidance_rate, guidance_type='classifier-free', label_dim=True).to(device)
            model_source = 'ldm'
    if net is None:
        raise ValueError("Got wrong settings: check dataset_name and guidance_type!")
    net.eval()

    return net, model_source

#----------------------------------------------------------------------------

@click.command()
# General options
@click.option('--solver',                  help='Solver type', metavar='STR',
              type=click.Choice(['euler', 'heun', 'dpm', 'ipndm', 'dpmpp', 'unipc', 'dyweight']), default='dyweight', show_default=True)
@click.option('--predictor_path',          help='Path to DyWeight predictor checkpoint (required if solver=dyweight)', metavar='PATH',
              type=str, default=None)
@click.option('--dataset_name',            help='Dataset name (required if solver != dyweight)', metavar='STR',
              type=str, default=None)
@click.option('--num_steps',               help='Number of sampling steps (ignored if solver=dyweight, loaded from ckpt)', metavar='INT',
              type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--guidance_type',           help='Guidance type (ignored if solver=dyweight, loaded from ckpt)', metavar='STR',
              type=click.Choice(['uncond', 'cg', 'cfg']), default='uncond', show_default=True)
@click.option('--guidance_rate',           help='Guidance rate (ignored if solver=dyweight, loaded from ckpt)', metavar='FLOAT',
              type=float, default=1.0, show_default=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',
              type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',
              type=parse_int_list, default='0-63', show_default=True)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',
              type=str)
@click.option('--use_fp16',                help='Whether to use mixed precision', metavar='BOOL',
              type=bool, default=False)

# Options for standard solvers (ignored if solver=dyweight, all loaded from checkpoint instead)
@click.option('--afs',                     help='Whether to use Analytical First Step', metavar='BOOL',
              type=bool, default=False, show_default=True)
@click.option('--denoise_to_zero',         help='Whether to denoise to zero', metavar='BOOL',
              type=bool, default=False, show_default=True)
@click.option('--max_order',               help='Max order for multistep solvers', metavar='INT',
              type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--predict_x0',              help='Whether to predict x0 (for dpmpp)', metavar='BOOL',
              type=bool, default=True, show_default=True)
@click.option('--variant',                 help='UniPC variant (bh1, bh2)', metavar='STR',
              type=click.Choice(['bh1', 'bh2']), default='bh2', show_default=True)
@click.option('--lower_order_final',       help='Whether to use lower order final (for dpmpp)', metavar='BOOL',
              type=bool, default=True, show_default=True)
@click.option('--schedule_type',           help='Schedule type',
              type=click.Choice(['polynomial', 'logsnr', 'time_uniform', 'discrete']), default='polynomial', show_default=True)
@click.option('--schedule_rho',            help='Schedule rho', metavar='FLOAT',
              type=float, default=7.0, show_default=True)

# Options for saving
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',
              type=str)
@click.option('--grid',                    help='Whether to make grid',
              type=bool, default=False)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',
              type=bool, default=True, is_flag=True)

#----------------------------------------------------------------------------

def main(solver, predictor_path, dataset_name, num_steps, guidance_type, guidance_rate,
         max_batch_size, seeds, grid, outdir, subdirs,
         afs, denoise_to_zero, max_order, predict_x0, lower_order_final, schedule_type, schedule_rho,
         device=torch.device('cuda'), **solver_kwargs):

    if solver == 'dyweight':
        if predictor_path is None:
            raise click.UsageError("--predictor_path is required when solver=dyweight")
    else:
        if dataset_name is None:
            raise click.UsageError("--dataset_name is required when solver is not dyweight")

    dist.init()

    try:
        num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

        if dist.get_rank() != 0:
            torch.distributed.barrier()     # rank 0 goes first

        prompt = solver_kwargs.pop('prompt', None)
        solver_kwargs = {key: value for key, value in solver_kwargs.items() if value is not None}
        solver_kwargs['solver'] = solver
        solver_kwargs['prompt'] = prompt

        if solver == 'dyweight':
            # Accept either a short experiment ID (e.g. "42") or a full .pkl path
            if not predictor_path.endswith('pkl'):
                predictor_path_str = '0' * (5 - len(predictor_path)) + predictor_path
                for file_name in os.listdir("./exps"):
                    if file_name.split('-')[0] == predictor_path_str:
                        file_list = [f for f in os.listdir(os.path.join('./exps', file_name))
                                     if f.endswith("pkl") and f.startswith("network-snapshot")]
                        max_index = -1
                        max_file = None
                        for ckpt_name in file_list:
                            try:
                                file_index = int(ckpt_name.split("-")[-1].split(".")[0])
                                if file_index > max_index:
                                    max_index = file_index
                                    max_file = ckpt_name
                            except ValueError:
                                continue
                        predictor_path = os.path.join('./exps', file_name, max_file)
                        break

            dist.print0(f'Loading DyWeight predictor from "{predictor_path}"...')
            with dnnlib.util.open_url(predictor_path, verbose=(dist.get_rank() == 0)) as f:
                checkpoint_data = pickle.load(f)

            predictor = checkpoint_data['predictor'].to(device)
            loss_fn = checkpoint_data['loss_fn']

            with torch.no_grad():
                hybrid_weights, t_scale = predictor()

            # All solver settings are loaded from the checkpoint
            solver_kwargs['hybrid_weights'] = hybrid_weights
            solver_kwargs['t_scale'] = t_scale
            solver_kwargs['num_steps'] = predictor.num_steps
            solver_kwargs['afs'] = predictor.afs
            solver_kwargs['max_order'] = loss_fn.max_order if hasattr(loss_fn, 'max_order') else 4
            solver_kwargs['predict_x0'] = loss_fn.predict_x0 if hasattr(loss_fn, 'predict_x0') else True
            solver_kwargs['lower_order_final'] = loss_fn.lower_order_final if hasattr(loss_fn, 'lower_order_final') else True
            solver_kwargs['schedule_type'] = loss_fn.schedule_type if hasattr(loss_fn, 'schedule_type') else 'polynomial'
            solver_kwargs['schedule_rho'] = loss_fn.schedule_rho if hasattr(loss_fn, 'schedule_rho') else 7.0

            dataset_name = (getattr(predictor, 'dataset_name', None)
                            or getattr(loss_fn, 'dataset_name', None)
                            or 'cifar10')
            solver_kwargs['dataset_name'] = dataset_name
            guidance_type = getattr(predictor, 'guidance_type', None)
            guidance_rate = getattr(predictor, 'guidance_rate', None)
            solver_kwargs['guidance_type'] = guidance_type
            solver_kwargs['guidance_rate'] = guidance_rate

        else:
            # Standard solvers: use CLI arguments directly
            solver_kwargs['num_steps'] = num_steps
            solver_kwargs['guidance_type'] = guidance_type
            solver_kwargs['guidance_rate'] = guidance_rate
            solver_kwargs['afs'] = afs
            solver_kwargs['denoise_to_zero'] = denoise_to_zero
            solver_kwargs['max_order'] = max_order
            solver_kwargs['predict_x0'] = predict_x0
            solver_kwargs['lower_order_final'] = lower_order_final
            solver_kwargs['schedule_type'] = schedule_type
            solver_kwargs['schedule_rho'] = schedule_rho
            solver_kwargs['dataset_name'] = dataset_name

        # Load pre-trained diffusion model
        net, solver_kwargs['model_source'] = create_model(dataset_name, guidance_type, guidance_rate, device)

        if dist.get_rank() == 0:
            torch.distributed.barrier()     # other ranks follow

        solver_kwargs['sigma_min'] = net.sigma_min
        solver_kwargs['sigma_max'] = net.sigma_max

        # Compute NFE (Number of Function Evaluations)
        _num_steps = solver_kwargs['num_steps']
        _afs = solver_kwargs.get('afs', False)
        _denoise_to_zero = solver_kwargs.get('denoise_to_zero', False)
        if solver == 'dyweight':
            nfe = (_num_steps - 1) if not _afs else (_num_steps - 2)
        elif solver in ['euler', 'ipndm', 'unipc', 'dpmpp']:
            nfe = _num_steps if _denoise_to_zero else _num_steps - 1
            if _afs:
                nfe -= 1
        elif solver in ['heun', 'dpm']:
            # heun and dpm both evaluate the model twice per step
            nfe = 2 * _num_steps if _denoise_to_zero else 2 * (_num_steps - 1)
            if _afs:
                nfe -= 1
        else:
            nfe = _num_steps - 1
        if dataset_name in ['ms_coco']:
            nfe = 2 * nfe   # classifier-free guidance doubles NFE
        solver_kwargs['nfe'] = nfe

        # Load MS-COCO prompts for FID-30k evaluation
        # Uses the 30k captions selected from https://github.com/boomb0om/text2image-benchmark
        sample_captions = []
        if dataset_name in ['ms_coco'] and solver_kwargs['prompt'] is None:
            prompt_path, _ = check_file_by_key('prompts')
            with open(prompt_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    sample_captions.append(row['text'])

        sampler_fn = get_solver_fn(solver_kwargs['solver'])

        # Print solver settings
        _skip_keys = {'model_source', 'use_fp16', 'return_inters'}
        dist.print0()
        dist.print0('=' * 50)
        dist.print0(f'  Solver: {solver.upper()}  |  NFE: {nfe}  |  Steps: {_num_steps}')
        dist.print0('=' * 50)
        dist.print0(f'  Dataset:       {dataset_name}')
        dist.print0(f'  Schedule:      {solver_kwargs.get("schedule_type")} (rho={solver_kwargs.get("schedule_rho")})')
        dist.print0(f'  Sigma:         [{solver_kwargs.get("sigma_min")} .. {solver_kwargs.get("sigma_max")}]')
        if solver_kwargs.get('afs'):
            dist.print0(f'  AFS:           True')
        if solver_kwargs.get('denoise_to_zero'):
            dist.print0(f'  Denoise to 0:  True')
        if solver in ['heun', 'dpm']:
            dist.print0(f'  NFE note:      2x steps (double evaluation per step)')
        if solver in ['dpmpp', 'unipc']:
            dist.print0(f'  Max order:     {solver_kwargs.get("max_order")}')
            dist.print0(f'  Predict x0:    {solver_kwargs.get("predict_x0")}')
            dist.print0(f'  Lower final:   {solver_kwargs.get("lower_order_final")}')
        if solver == 'unipc':
            dist.print0(f'  Variant:       {solver_kwargs.get("variant", "bh2")}')
        if solver_kwargs.get('guidance_type') and solver_kwargs['guidance_type'] != 'uncond':
            dist.print0(f'  Guidance:      {solver_kwargs["guidance_type"]} (rate={solver_kwargs.get("guidance_rate")})')
        if dataset_name in ['ms_coco'] and solver_kwargs.get('prompt'):
            dist.print0(f'  Prompt:        {solver_kwargs["prompt"]}')
        if solver == 'dyweight':
            hw = solver_kwargs.get('hybrid_weights')
            ts = solver_kwargs.get('t_scale')
            if hw is not None:
                dist.print0(f'  Hybrid weights ({hw.shape[0]} steps):')
                for i in range(hw.shape[0]):
                    row = hw[i, :i+1]
                    dist.print0(f'    Step {i}: [{"/".join(f"{w:.4f}" for w in row)}]')
            if ts is not None:
                dist.print0(f'  t_scale:       [{", ".join(f"{s:.4f}" for s in ts.tolist())}]')
        dist.print0('-' * 50)

        if outdir is None:
            base = "./samples/grids" if grid else "./samples"
            outdir = os.path.join(base, dataset_name, f"{solver}_nfe{nfe}")
        dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')

        for batch_id, batch_seeds in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
            torch.distributed.barrier()
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)

            class_labels = c = uc = None
            if net.label_dim:
                if solver_kwargs['model_source'] == 'adm':
                    class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device)
                elif solver_kwargs['model_source'] == 'ldm' and dataset_name == 'ms_coco':
                    if solver_kwargs['prompt'] is None:
                        prompts = sample_captions[batch_seeds[0]:batch_seeds[-1]+1]
                    else:
                        prompts = [solver_kwargs['prompt'] for _ in range(batch_size)]
                    if solver_kwargs['guidance_rate'] != 1.0:
                        uc = net.model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = net.model.get_learned_conditioning(prompts)
                else:
                    class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]

            with torch.no_grad():
                if solver_kwargs['model_source'] == 'ldm':
                    with autocast("cuda"):
                        with net.model.ema_scope():
                            images = sampler_fn(net, latents, condition=c, unconditional_condition=uc, **solver_kwargs)
                            images = net.model.decode_first_stage(images)
                else:
                    images = sampler_fn(net, latents, class_labels=class_labels, **solver_kwargs)

            if grid:
                images = torch.clamp(images / 2 + 0.5, 0, 1)
                os.makedirs(outdir, exist_ok=True)
                nrows = int(images.shape[0] ** 0.5)
                image_grid = make_grid(images, nrows, padding=0)
                save_image(image_grid, os.path.join(outdir, "grid.png"))
            else:
                images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for seed, image_np in zip(batch_seeds, images_np):
                    image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f'{seed:06d}.png')
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        torch.distributed.barrier()
        dist.print0('Done.')

    finally:
        dist.cleanup()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
