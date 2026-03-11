import pdb
import os, sys
import torch
import torch.distributed as dist
from tqdm import tqdm
from models.flux.dyweight_pipeline_flux import DyWeight_FluxPipeline
from models.flux.flow_dpmsolver_multistep import DPMSolverMultistepScheduler
from models.flux.config import InferBasicCfg, load_infer_cfg
from models.flux.io_utils import read_prompts, print_with_prefix

def shard_prompt_indices(num_prompts: int, world_size: int, rank: int):
    padded = ((num_prompts + world_size - 1) // world_size) * world_size
    idxs = torch.arange(padded, dtype=torch.long)
    idxs[idxs >= num_prompts] = -1
    shards = idxs.tensor_split(world_size)
    return [int(i) for i in shards[rank].tolist() if int(i) >= 0]

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def load_dyweight_ckpt(path, device="cuda", dtype=torch.bfloat16):
    obj = torch.load(path, map_location="cpu")
    W = obj["dyweight_weights_raw"]
    win = obj["window_size"]
    scale = obj["dyweight_t_scales"]
    return W.to(device), win, scale.to(device)

def build_pipeline(cfg, cuda_device):
    pipe = DyWeight_FluxPipeline.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    if cfg.multistep_mode == "dpmpp":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_flow_sigmas=True, prediction_type="flow_prediction")
    pipe.to(f"cuda:{cuda_device}")
    return pipe

@torch.inference_mode()
def main():
    assert len(sys.argv) == 2 and sys.argv[1].endswith((".yml", ".yaml")), "usage: torchrun ... sample_flux.py <cfg.yaml>"
    cfg: InferBasicCfg = load_infer_cfg(sys.argv[1])

    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)
    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_device)
    is_main = (rank==0)

    prompts = read_prompts(cfg.dataset_path) # prompts loading
    N = len(prompts); spp = int(cfg.samples_per_prompt); B = int(cfg.batch_size)
    dyweight_W = None; dyweight_scale = None 
    if cfg.multistep_mode == "dyweight":
        assert cfg.dyweight_ckpt_path is not None
        dyweight_W, win, dyweight_scale = load_dyweight_ckpt(cfg.dyweight_ckpt_path)

    out_root = os.path.join(cfg.out_dir, f"{cfg.dataset_name}_{N}x{spp}@{cfg.multistep_mode}_nfe{int(cfg.steps)}")
    if is_main:
        os.makedirs(out_root, exist_ok=True)
        print_with_prefix(f"Save .png samples in {out_root}")

    pipe = build_pipeline(cfg, cuda_device=local_device)
    dist.barrier()

    my_indices = shard_prompt_indices(num_prompts=N, world_size=world_size, rank=rank)
    pbar = tqdm(total=len(my_indices)*spp, ncols=100, disable=not is_main, desc=f"rank0 sampling")
    for k in range(int(cfg.samples_per_prompt)):
        for batch_idx in chunked(list(my_indices), B):
            prompts_batch = [prompts[i] for i in batch_idx]
            generators = [torch.Generator(device="cpu").manual_seed(int(cfg.seed)+i+k*N) for i in batch_idx]
            imgs = pipe(
                prompts_batch,
                num_inference_steps=int(cfg.steps), guidance_scale=float(cfg.guidance_scale),
                height=int(cfg.height), width=int(cfg.width), generator=generators,
                t_scales=dyweight_scale, weights=dyweight_W, step_mode=cfg.multistep_mode,
                ipndm_max_order=int(cfg.ipndm_order),
            ).images
            for img, idx0 in zip(imgs, batch_idx): # save
                img.save(os.path.join(out_root, f"{idx0:06d}_{k:02d}.png"))
            if is_main: pbar.update(len(batch_idx))
    if is_main: pbar.close()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()