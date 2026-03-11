import os
import pdb
import json
import math
import lpips
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from time import time, strftime
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast, CLIPImageProcessor, CLIPVisionModelWithProjection

from models.flux.config import load_train_cfg
from models.flux.dyweight import DyweightParams, DyweightRunner
from models.flux.sampling import get_noise, prepare_embeds, get_schedule, unpack, vae_decode, vae_decode_tensor
from models.flux.io_utils import create_logger, make_png2gif, plot_loss_fig, visual_teacher_student, plot_weights_trend

# Datasets
def read_metadatas(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]
class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        self.metadatas = read_metadatas(str(self.path))
    def __len__(self) -> int:
        return len(self.metadatas)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.metadatas[idx]
        prompt = meta.get("prompt", "")
        return {
            "prompt": prompt,
            "index": idx,
        }
def prompt_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompts = [b["prompt"] for b in batch]
    return {
        "prompt": prompts,
    }

def _to_fp32_cpu_state(sd):
    return {k: v.detach().to(torch.float32).cpu() for k, v in sd.items()}

@torch.no_grad()
def materialize_dyweight(unwrapped_runner):
    rp = unwrapped_runner.dyweight_params
    K  = rp.K
    W_raw  = torch.zeros(K, K, dtype=torch.float32)
    t_scales = torch.zeros(K, dtype=torch.float32)
    for i in range(K):
        w_raw, start  = rp.weights_row(i)
        w_raw = w_raw.detach().float().cpu()
        W_raw[i,  start:i+1] = w_raw
        t_scales[i] = rp.t_scale(i).detach().float().cpu()
    meta = {
        "K": int(K),
        "tscale_low": float(rp.tscale_low),
        "tscale_high": float(rp.tscale_high),
    }
    return W_raw, t_scales, meta

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML Path")
    return p.parse_args()
    
def main():
    # load config
    args = parse_args()
    cfg = load_train_cfg(args.config)
    # accelerator init
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,)
    # Set up experiment folders
    str_time = strftime('%Y%m%d-%H%M%S') # can be added to exp_name
    loss_type = str(cfg.loss).lower(); lr_rate=f"{float(cfg.learning_rate):.0e}".replace("+0", "").replace("+", "")
    mb=int(cfg.train_batch_size); grad_acc=int(cfg.gradient_accumulation_steps)
    exp_name = f"{cfg.dyweight_exp_name}@E{cfg.num_train_epochs}_B{mb}x{grad_acc}_hw{cfg.height}x{cfg.width}_nfe{cfg.teacher_steps}x{cfg.student_steps}_{loss_type}{lr_rate}"
    exp_dir = os.path.join(cfg.output_dir, exp_name)
    ckpt_dir = os.path.join(exp_dir, f"checkpoints")
    vis_images_dir = os.path.join(exp_dir, f"vis_images")
    vis_weights_dir = os.path.join(exp_dir, f"vis_weights")
    if accelerator.is_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(vis_weights_dir, exist_ok=True)
        os.makedirs(vis_images_dir, exist_ok=True)
        log_loss_list = []
    logger = create_logger(exp_dir, accelerator)
    # loader models
    flux_path = cfg.model_id
    euler_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(flux_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(flux_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(flux_path, subfolder="text_encoder")
    tokenizer_2 = T5TokenizerFast.from_pretrained(flux_path, subfolder="tokenizer_2")
    text_encoder_2 = T5EncoderModel.from_pretrained(flux_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(flux_path, subfolder="vae")
    vae.disable_slicing(); vae.disable_tiling() # critical
    transformer = FluxTransformer2DModel.from_pretrained(flux_path, subfolder="transformer")
    image_processor = VaeImageProcessor(vae_scale_factor=16)
    # freeze parameters
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype) # keep the VAE in float32 when train.
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    # ----- Data training datasets prepare & epochs confirm -----
    train_dataset = PromptDataset(cfg.dataset_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=prompt_collate,
        batch_size=cfg.train_batch_size, num_workers=cfg.dataloader_num_workers, drop_last=True,)
    if cfg.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        updates_per_epoch = math.ceil(len_train_dataloader_after_sharding / cfg.gradient_accumulation_steps)
        num_training_steps_for_scheduler = cfg.num_train_epochs * updates_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = cfg.max_train_steps * accelerator.num_processes
    # ----- training runner & optimizer params & loss & lr_scheduler -----
    dyweight_params = DyweightParams(
        K=cfg.student_steps, init_mode=cfg.dyweight_init, window_size=cfg.dyweight_window,
    ).to(accelerator.device)
    dyweight_runner = DyweightRunner(transformer=transformer, dyweight_params=dyweight_params)
    if loss_type == "lpips":
        loss_func = lpips.LPIPS(net='vgg').to(accelerator.device)
    if loss_type in ("mse", "l2"):
        loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.RMSprop(dyweight_runner.dyweight_params.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay),)
    optimizer = torch.optim.AdamW(dyweight_runner.dyweight_params.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay),)
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=(cfg.lr_warmup_steps),
        num_training_steps=num_training_steps_for_scheduler,)
    
    dyweight_runner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(dyweight_runner, optimizer, train_dataloader, lr_scheduler)
    
    # ----- for DDP -----
    updates_per_epoch_real = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * updates_per_epoch_real
        if num_training_steps_for_scheduler != cfg.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"train_dataloader after prepare = {len(train_dataloader)} != expected {len_train_dataloader_after_sharding} before prepare; LR scheduler total steps may be off."
            )
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / updates_per_epoch_real)
    
    logger.info(" ***** Running DyWeight training ***** ")
    logger.info(f"  Num examples = {len(train_dataset)} Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Total train batch size (w. distributed & micro-batch & accumulation) = {accelerator.num_processes}x{cfg.train_batch_size}x{cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}, Loss type={loss_type}")
    logger.info(f"  Teacher Steps={cfg.teacher_steps}, Student Steps={cfg.student_steps}")
    logger.info(f"  Training Result = {exp_dir}")
    progress_bar = tqdm(
        range(0, int(cfg.max_train_steps)),
        initial=0, desc="Steps", disable=not accelerator.is_main_process,)
    
    start_time = time()
    global_step = 0
    acc_loss_sum = torch.tensor(0.0, device=accelerator.device)
    acc_n = torch.tensor(0, device=accelerator.device, dtype=torch.long)
    for epoch in range(0, cfg.num_train_epochs):
        for idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dyweight_runner):
                bsz = len(batch["prompt"])
                noise = get_noise(bsz, cfg.height, cfg.width, accelerator.device, weight_dtype) # torch.randn is also ok
                image_seq_len = (noise.shape[-1]*noise.shape[-2])//4
                with torch.no_grad(): # prepare text condition
                    inp = prepare_embeds(
                        img=noise, prompt=batch["prompt"], guidance=float(cfg.guidance_scale),
                        clip_encoder=text_encoder, t5_encoder=text_encoder_2, 
                        clip_tokenizer=tokenizer, t5_tokenizer=tokenizer_2)
                with torch.no_grad(): # teacher sample
                    teacher_sigmas = get_schedule(int(cfg.teacher_steps), image_seq_len=image_seq_len,) # float
                    teacher_latents = dyweight_runner(
                        inp["img"], teacher_sigmas, guidance=inp["guidance"], pooled_prompt_embeds=inp["pooled_prompt_embeds"], 
                        prompt_embeds=inp["prompt_embeds"], text_ids=inp["text_ids"], image_ids=inp["img_ids"],
                        mode="teacher")
                # student sample
                student_sigmas = get_schedule(int(cfg.student_steps), image_seq_len=image_seq_len,) # float
                student_latents = dyweight_runner(
                    inp["img"], student_sigmas, guidance=inp["guidance"], pooled_prompt_embeds=inp["pooled_prompt_embeds"], 
                    prompt_embeds=inp["prompt_embeds"], text_ids=inp["text_ids"], image_ids=inp["img_ids"],
                    mode="student")
                if loss_type == "lpips":
                    teacher_tensors = vae_decode_tensor(unpack(teacher_latents, cfg.height, cfg.width), vae)
                    student_tensors = vae_decode_tensor(unpack(student_latents, cfg.height, cfg.width), vae)
                    loss = loss_func(teacher_tensors.float(), student_tensors.float())
                else:
                    loss = loss_func(teacher_latents.float(), student_latents.float())
                acc_loss_sum += loss.detach() * bsz
                acc_n += bsz
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                tot_loss_sum = accelerator.gather_for_metrics(acc_loss_sum).sum()
                tot_n = accelerator.gather_for_metrics(acc_n).sum()
                upd_loss = (tot_loss_sum / tot_n).item()
                acc_loss_sum.zero_(); acc_n.zero_()
                # visual weights every step
                if accelerator.is_main_process and global_step%1==0:
                    out_png = os.path.join(vis_weights_dir, f"step{global_step:06d}.png")
                    unwrapped = accelerator.unwrap_model(dyweight_runner)
                    plot_weights_trend(unwrapped.dyweight_params, out_png)
                # visual teacher and student image
                if accelerator.is_main_process and (global_step % int(cfg.visual_steps) == 0): 
                    with torch.no_grad():
                        teacher_samples = unpack(teacher_latents, cfg.height, cfg.width).detach()
                        student_samples = unpack(student_latents, cfg.height, cfg.width).detach()
                        teacher_img = vae_decode(teacher_samples, vae, image_processor,)
                        student_img = vae_decode(student_samples, vae, image_processor,)
                    visual_teacher_student(teacher_img, student_img, os.path.join(vis_images_dir, f"TS_{global_step:06d}.png"))
                if global_step % cfg.ckpt_save_steps == 0: # save checkpoints
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(dyweight_runner)
                    W_raw, t_scales, dyweight_meta = materialize_dyweight(unwrapped)
                    ckpt = {
                        "window_size":              cfg.dyweight_window,
                        "dyweight_weights_raw":     W_raw,       # [K,K]
                        "dyweight_t_scales":        t_scales,    # [K]
                    }
                    if accelerator.is_main_process:
                        accelerator.save(ckpt, os.path.join(ckpt_dir, f"step{global_step:06d}.pt"))    
                if accelerator.is_main_process:
                    logs = {"step_loss": upd_loss, "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    log_loss_list.append(upd_loss)
                    # accelerator.log(logs, step=global_step)
                    if global_step%16 == 0: # plot loss trend
                        plot_loss_fig(exp_dir, log_loss_list)
                if global_step >= cfg.max_train_steps:
                    break
    progress_bar.close()
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(dyweight_runner)
    W_raw, t_scales, dyweight_meta = materialize_dyweight(unwrapped)
    ckpt = {
        "window_size":              cfg.dyweight_window,
        "dyweight_weights_raw":     W_raw,       # [K,K]
        "dyweight_t_scales":        t_scales,    # [K]
    }
    if accelerator.is_main_process:
        accelerator.save(ckpt, os.path.join(exp_dir, f"final-step.pt"))
        # accelerator.save_state(os.path.join(exp_dir, f"accelerate-state-step{global_step:06d}"))
        make_png2gif(vis_weights_dir, os.path.join(exp_dir, "weights_evolution.gif"))
        plot_loss_fig(exp_dir, log_loss_list)
    logger.info(f"***** After {time() - start_time:.3f}s end training *****")
    accelerator.end_training()

if __name__ == "__main__":
    main()