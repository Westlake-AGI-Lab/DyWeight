import os
import json
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
def load_prompts(path: str):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l)["prompt"] for l in f if l.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--prompts_file", default="dyweight_flux/dataset/eval/drawbench.txt")
    ap.add_argument("--clip_ckpt", default="path_tp/openai-clip-vit-base-patch16")
    args = ap.parse_args()
    device = "cuda"
    model = CLIPModel.from_pretrained(args.clip_ckpt).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.clip_ckpt)
    prompts = load_prompts(args.prompts_file)
    files = sorted([
        f for f in os.listdir(args.gen_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    assert len(files) > 0, f"No image files found in {args.gen_dir}"
    score_sum = 0.0
    count = 0
    # cache text features by prompt_idx (important when samples_per_prompt > 1)
    text_feat_cache = {}
    with torch.no_grad():
        for fn in tqdm(files, desc="clipscore"):
            # expected: 000123_01.png
            stem = os.path.splitext(fn)[0]
            prompt_idx = int(stem.split("_")[0])
            assert 0 <= prompt_idx < len(prompts), f"prompt_idx out of range: {prompt_idx}, file={fn}"
            if prompt_idx not in text_feat_cache:
                prompt = prompts[prompt_idx]
                ti = proc(
                    text=prompt, return_tensors="pt",
                    truncation=True, padding=False,
                    max_length=proc.tokenizer.model_max_length,).to(device)
                tfeat = F.normalize(model.get_text_features(**ti), dim=-1)
                text_feat_cache[prompt_idx] = tfeat
            tfeat = text_feat_cache[prompt_idx]
            img = Image.open(os.path.join(args.gen_dir, fn)).convert("RGB")
            ii = proc(images=img, return_tensors="pt").to(device)
            ifeat = F.normalize(model.get_image_features(**ii), dim=-1)
            score = float(torch.matmul(ifeat, tfeat.T).squeeze().item())
            score_sum += score
            count += 1
    mean_score = score_sum / count
    print(f"[CLIPScore] path={args.gen_dir}")
    print(f"[CLIPScore] count={count} mean={100.0 * mean_score:.6f}")
if __name__ == "__main__":
    main()