import os, json
from dataclasses import dataclass, field, is_dataclass, asdict
from typing import List, Optional, Dict
import torch
import pandas as pd
from omegaconf import OmegaConf
import yaml

def load_dataclass_cfg(path: str, cls):
    base = OmegaConf.structured(cls)
    file = OmegaConf.load(path)
    merged = OmegaConf.merge(base, file)
    return OmegaConf.to_object(merged)

@dataclass
class InferBasicCfg:
    model_id: str = "black-forest-labs/FLUX.1-dev"
    steps: int = 28
    seed: int = 42
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 3.5
    samples_per_prompt: int = 1
    batch_size: int = 1
    dataset_path: str = ""
    dataset_name: str = ""
    out_dir: str = "./output_eval"
    # dyweight
    multistep_mode: str = "dyweight" # "dyweight" | "ipndm" | "dpmpp"
    dyweight_ckpt_path: Optional[str] = None
    ipndm_order: int=3
def load_infer_cfg(path: str) -> InferBasicCfg:
    return load_dataclass_cfg(path, InferBasicCfg)

@dataclass
class TrainingCfg:
    model_id: str = "black-forest-labs/FLUX.1-dev"
    # basic
    output_dir: str = "./training_output"
    mixed_precision: str = "bf16"
    report_to: str = "wandb"
    # dataset & dataloader
    dataset_path: str = "dataset/train/val2014_N5000.jsonl"
    train_batch_size: int = 1
    dataloader_num_workers: int = 4
    # optimizer
    gradient_accumulation_steps: int = 4
    optimizer: str="adamw"
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    # loss function | learning scheduler
    loss: str="mse" # lpips | mse
    learning_rate: float = 5e-4
    lr_scheduler_type: str = "cosine" #Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_warmup_steps: int = 0
    lr_num_cycles: int = 1
    lr_power: int = 1
    # training & checkpoints
    max_train_steps: Optional[int] = None
    num_train_epochs: int = 1
    ckpt_save_steps: int = 64
    visual_steps: int = 32
    # teacher / scheduler
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 3.5
    teacher_steps: int = 28
    student_steps: int = 7
    # For dyweights training
    dyweight_init: str="ipndm" #"ipndm" "euler" "zeros" "uniform"
    dyweight_window: int=3
    dyweight_exp_name: Optional[str] = "mscoco5k"
def load_train_cfg(path: str) -> TrainingCfg:
    return load_dataclass_cfg(path, TrainingCfg)