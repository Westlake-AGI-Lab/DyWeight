#!/bin/bash

####################################
### A. Train DyWeight Predictor  ###
####################################

# Hardware Note: 
# - Use A100 for LSUN_Bedroom_ldm & Stable Diffusion
# - Use RTX4090 for other experiments
# - Adjust batch size according to your GPU memory

# Note on num_steps:
# - Sampling steps = num_steps (N)
# - NFE = (N-1)-1 if afs=True, else (N-1)
# - NFE doubles for ms_coco (classifier-free guidance)

train_dyweight() {
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 --master_port=11111 \
        train.py \
        --dataset_name="$1" \
        --batch="$2" \
        --total_kimg="$3" \
        $SOLVER_FLAGS \
        $SCHEDULE_FLAGS \
        $ADDITIONAL_FLAGS \
        $GUIDANCE_FLAGS \
        $LOSS_FLAGS \
        $LR_FLAGS
}

## A.1 CIFAR-10 ##
echo "Training DyWeight on cifar10..."
SOLVER_FLAGS="--sampler_stu=dyweight --sampler_tea=ipndm --num_steps=5 --afs=True --max_history_steps=3"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--teacher_steps=35 --use_warmup=True --warmup_ratio=0.1"
GUIDANCE_FLAGS=""
LOSS_FLAGS="--loss_type=inception"
LR_FLAGS="--lr=3e-2 --use_cosine_annealing=True --cosine_min_lr_ratio=1e-2"
train_dyweight "cifar10" 16 10

## A.2 ImageNet-64 ##
# echo "Training DyWeight on imagenet64..."
# SOLVER_FLAGS="--sampler_stu=dyweight --sampler_tea=ipndm --num_steps=5 --afs=True --max_history_steps=3"
# SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
# ADDITIONAL_FLAGS="--teacher_steps=35 --use_warmup=True --warmup_ratio=0.1"
# GUIDANCE_FLAGS=""
# LOSS_FLAGS="--loss_type=inception"
# LR_FLAGS="--lr=5e-3 --use_cosine_annealing=True --cosine_min_lr_ratio=1e-2"
# train_dyweight "imagenet64" 16 10

## A.3 FFHQ ##
# echo "Training DyWeight on ffhq..."
# SOLVER_FLAGS="--sampler_stu=dyweight --sampler_tea=ipndm --num_steps=5 --afs=True --max_history_steps=3"
# SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
# ADDITIONAL_FLAGS="--teacher_steps=35 --use_warmup=True --warmup_ratio=0.1"
# GUIDANCE_FLAGS=""
# LOSS_FLAGS="--loss_type=inception"
# LR_FLAGS="--lr=1e-2 --use_cosine_annealing=True --cosine_min_lr_ratio=1e-2"
# train_dyweight "ffhq" 16 10

## A.4 AFHQ v2 ##
# echo "Training DyWeight on afhqv2..."
# SOLVER_FLAGS="--sampler_stu=dyweight --sampler_tea=ipndm --num_steps=5 --afs=True --max_history_steps=3"
# SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
# ADDITIONAL_FLAGS="--teacher_steps=35 --use_warmup=True --warmup_ratio=0.1"
# GUIDANCE_FLAGS=""
# LOSS_FLAGS="--loss_type=inception"
# LR_FLAGS="--lr=1e-2 --use_cosine_annealing=True --cosine_min_lr_ratio=1e-2"
# train_dyweight "afhqv2" 16 10

## A.4 LSUN Bedroom (LDM) ##
# echo "Training DyWeight on lsun_bedroom_ldm..."
# SOLVER_FLAGS="--sampler_stu=dyweight --sampler_tea=ipndm --num_steps=5 --afs=True --max_history_steps=3"
# SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
# ADDITIONAL_FLAGS="--teacher_steps=35 --use_warmup=True --warmup_ratio=0.1"
# GUIDANCE_FLAGS="--guidance_type=uncond --guidance_rate=1"
# LOSS_FLAGS="--loss_type=inception"
# LR_FLAGS="--lr=1e-2 --use_cosine_annealing=True --cosine_min_lr_ratio=1e-2"
# train_dyweight "lsun_bedroom_ldm" 4 10

## A.5 Stable Diffusion (MS-COCO) ##
# echo "Training DyWeight on ms_coco..."
# SOLVER_FLAGS="--sampler_stu=dyweight --sampler_tea=ipndm --num_steps=5 --afs=False --max_history_steps=2"
# SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
# ADDITIONAL_FLAGS="--max_order=2 --lower_order_final=True --teacher_steps=35 --use_warmup=True --warmup_ratio=0.1"
# GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
# LOSS_FLAGS="--loss_type=l2"
# LR_FLAGS="--lr=5e-4 --use_cosine_annealing=True --cosine_min_lr_ratio=1e-2"
# train_dyweight "ms_coco" 2 5

####################################
### B. Generate Samples for FID  ###
####################################

# Trained predictors are saved in ./exps/ (5-digit experiment numbers)
# Use either:
#   --predictor_path=/full/path
#   --predictor_path=EXP_NUMBER (e.g., 00000)

generate_samples() {
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 --master_port=22222 \
        sample.py \
        --predictor_path="$1" \
        --batch="$2" \
        --seeds="$3" \
        $SAMPLE_FLAGS
}

## B.1 Standard Datasets (50k samples for FID) ##
echo "Generating samples..."
SAMPLE_FLAGS="--solver=dyweight"
generate_samples "00000" 128 "0-49999"


# echo "Generating samples..."
# SAMPLE_FLAGS="--solver=ipndm --dataset_name=cifar10 --num_steps=5 --afs=True"
# generate_samples "" 256 "0-49999"

## B.2 Stable Diffusion ##
# echo "Generating samples for ms_coco..."
# SAMPLE_FLAGS=""
# generate_samples "00000" 8 "0-29999"

####################################
### C. Evaluation                ###
####################################

## C.1 FID Calculation ##
python -m eval.fid calc \
    --images="samples/cifar10/dyweight_nfe3" \
    --ref="ref/cifar10-32x32.npz"


## C.2 CLIP Score Calculation ##
# torchrun --standalone --nproc_per_node=2 eval/clip_score.py calc \
#     --images="samples/ms_coco/dyweight_nfe8" 
