import os
import json
import random
from pathlib import Path

import numpy as np

# =========================
# User settings (edit here)
# =========================
IMG_DIR = f"path_to/mscoco/images/val2014"
ANN_PATH = f"path_to/mscoco/annotations/captions_val2014.json"

OUT_ROOT = "path_to/dyweight_flux/dataset/eval"
N_SAMPLES = 30000
SEED = 2024

PROMPT_JSONL_OUT = f"{OUT_ROOT}/coco30k_val2014_seed{SEED}_prompts.jsonl"
NPZ_OUT = f"{OUT_ROOT}/fid_ref/coco30k_val2014_pytorch-fid_stats_seed{SEED}.npz"

DEVICE = "cuda"
BATCH = 1        # COCO images have different sizes
NUM_WORKERS = 8

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(NPZ_OUT)=, exist_ok=True)
    with open(ANN_PATH, "r", encoding="utf-8") as f:
        ann = json.load(f)
    images = ann["images"]
    annotations = ann["annotations"]
    # Keep one caption per image_id (first occurrence)
    imgid2cap = {}
    for a in annotations:
        img_id = a["image_id"]
        if img_id not in imgid2cap:
            imgid2cap[img_id] = a["caption"]
    # Build valid pool: each item has (abs_path, prompt)
    pool = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for im in images:
        fn = im["file_name"]
        abs_path = os.path.abspath(os.path.join(IMG_DIR, fn))
        if not os.path.isfile(abs_path):
            continue
        if os.path.splitext(abs_path)[1].lower() not in valid_exts:
            continue
        prompt = str(imgid2cap.get(im["id"], "")).strip()
        if not prompt:
            continue
        pool.append({"abs_path": abs_path, "prompt": prompt})
    assert len(pool) >= N_SAMPLES, f"pool size {len(pool)} < requested {N_SAMPLES}"
    random.seed(SEED)
    idxs = list(range(len(pool)))
    random.shuffle(idxs)
    rows = [pool[i] for i in idxs[:N_SAMPLES]]
    # Save prompt JSONL (for generation)
    with open(PROMPT_JSONL_OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"prompt": r["prompt"]}, ensure_ascii=False) + "\n")
    # Compute FID real stats from sampled real images
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import get_activations
    files = [r["abs_path"] for r in rows]
    dims = 2048
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(DEVICE).eval()
    acts = get_activations(files, model, BATCH, dims, DEVICE, NUM_WORKERS)
    mu = acts.mean(0)
    sigma = np.cov(acts, rowvar=False)
    np.savez(NPZ_OUT, mu=mu, sigma=sigma)
    print(f"[FID] subset size     : {len(rows)}")
    print(f"[FID] prompt jsonl    : {PROMPT_JSONL_OUT}")
    print(f"[FID] real fid stats  : {NPZ_OUT}")
if __name__ == "__main__":
    main()