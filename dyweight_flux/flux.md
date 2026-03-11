## 1. Environment Setup

Create a clean conda environment and install the required dependencies:

```bash
conda create -n flux python=3.10.18
conda activate flux

pip install torch==2.3.1 torchvision deepspeed xformers==0.0.27 \
  diffusers==0.31.0 numpy==1.26.4 opencv-python-headless bitsandbytes==0.46.0 \
  einops gradio nvitop peft==0.15.2 onnxruntime-gpu==1.21.0 safetensors==0.5.3 \
  scipy==1.15.2 tensorboard transformers==4.50.3 wandb matplotlib protobuf omegaconf \
  sentencepiece scikit-learn seaborn hpsv2 image-reward pytorch-fid
```

## 2. Download checkpoints
Please download the official [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) weights from Hugging Face.

## 3. Inference with Released DyWeight Checkpoints
Example evaluation command:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port 29523 \
  sample_flux.py dyweight_flux/eval_example.yaml
```
Please update the config file `(configs/flux/eval_example.yaml)` to match your local paths.

We also provide evaluation scripts for computing metrics from generated images.

## 4. Train DyWeight on FLUX.1-dev
If you would like to train DyWeight yourself, you can do so with a data-free setup. 

By data-free, we mean that DyWeight training does **not** require paired real images or a full image dataset.  
Instead, training only needs a list of **text prompts**.

As discussed in the paper, DyWeight is trained with result-level supervision, while the optimization is still performed in the latent space of the diffusion model.  
In other words, we do not train on paired image annotations; we optimize the DyWeight parameters by matching the student sampling behavior to the teacher behavior under the same prompt/noise conditions.

Example training command:
```bash
export TOKENIZERS_PARALLELISM=false
accelerate launch \
         --num_machines 1 \
         --num_processes 8 \
         --num_cpu_threads_per_process 16 \
         --main_process_port 23456 \
         train_flux.py \
         --config "dyweight_flux/train_example.yaml"
```

You may also use other prompt sets for your own experiments.
For example, you can download MS-COCO captions from: http://images.cocodataset.org/annotations/annotations_trainval2014.zip. 
From the caption annotations, you can randomly sample different prompt-set sizes depending on your compute budget, e.g.:

- 500 prompts (quick test)

- 5k prompts (paper-scale setting)

- 10k prompts (larger prompt pool)

We provide a preprocessing script to convert prompts into the training dataset format used by this repository: `dyweight_flux/preprocess/build_prompt_jsonl_from_coco.py`


## 5. Evaluation

Evaluation is a critical part of our pipeline. In the paper, we report four metrics:

- **CLIPScore**
- **HPSv2.1**
- **ImageReward**
- **FID**

We evaluate on both DrawBench and MS-COCO.  
Although DyWeight is trained on MS-COCO prompts, we also evaluate on DrawBench, which further demonstrates generalization.

### HPSv2.1

Please download the required weights:

- [`clip-vit-h-14-laion2b-s32b-b79k/open_clip_pytorch_model.bin`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)
- [`HPS_v2.1_compressed.pt`](https://huggingface.co/xswu/HPSv2/tree/main)

For simplicity, we recommend directly editing the HPS source file to set the local checkpoint paths.

```python
# hpsv2/img_score.py

## line 22
- 'laion2B-s32B-b79K'
+ 'path_to/clip-vit-h-14-laion2b-s32b-b79k/open_clip_pytorch_model.bin'

## line 42
- def score(img_path: Union[list, str, Image.Image], prompt: str, cp: str = None, hps_version: str = "v2.0") -> list:
+ def score(img_path: Union[list, str, Image.Image], prompt: str, cp='path_to/HPS_v2.1_compressed.pt', hps_version: str = "v2.0") -> list:
```

Example training command:
```bash
python dyweight_flux/eval/hps_v2.py --gen_dir "path_to_your_RESULT_PATH"
```

### ImageReward
Please download the checkpoint from: [`zai-org/ImageReward`](https://huggingface.co/zai-org/ImageReward/tree/main)

Example training command:
```bash
python dyweight_flux/eval/imagereward.py --gen_dir "path_to_your_RESULT_PATH"
```

### CLIP score
Different CLIP backbones may produce different absolute CLIPScore values. Common choices include:
- [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16/tree/main)
- [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)

Example training command:
```bash
python dyweight_flux/eval/clipscore.py --gen_dir "path_to_your_RESULT_PATH"
```

### Preparing an MS-COCO subset for FID evaluation

FID (Fréchet Inception Distance) is a measure of similarity between two datasets of images. In our experiments, we use `pytorch_fid`.
For prompt-conditioned evaluation, we need two things:
1. A reference distribution computed from real images (saved as `.npz` statistics), and
2. A prompt set used to generate model outputs for comparison.

Therefore, the first step is to sample a subset from MS-COCO, then:
- compute and save the real-image FID statistics, and
- save the corresponding prompts as a JSONL file for generation.

In the paper, we use **30k** prompts/images. In general, using more samples leads to a more stable and reliable FID estimate (at higher compute cost). The code you can refer to `dyweight_flux/preprocess/prepare_coco_fid_subset.py`