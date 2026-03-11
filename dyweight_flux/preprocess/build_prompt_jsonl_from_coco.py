"""
Build prompt-only JSONL files from MS-COCO caption annotations.

Output format (one JSON per line):
{"prompt": "..."}
"""

import json
from pathlib import Path
import numpy as np

# =========================
# User settings (edit here)
# =========================
ANN_PATH = "path_to/mscoco_annotations/captions_train2014.json"
OUT_DIR = "dyweight_flux/dataset/train"
SEED = 42
SIZES = [500, 5000, 10000, 30000]
OUT_PREFIX = "train2014"  # e.g. "train2014"

def load_coco_captions(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions = []
    for ann in data["annotations"]:
        cap = str(ann.get("caption", "")).strip()
        if cap:
            captions.append(cap)
    return captions
def write_jsonl(path: Path, prompts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
def main():
    ann_path = Path(ANN_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = load_coco_captions(str(ann_path))
    raw_n = len(captions)
    total_n = len(captions)
    assert total_n > 0, "No valid captions found."
    assert max(SIZES) <= total_n, f"Requested {max(SIZES)} > available {total_n}"
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(total_n)
    prefix = OUT_PREFIX if OUT_PREFIX is not None else ann_path.stem
    # e.g. captions_train2014_N500_seed42.jsonl
    for n in SIZES:
        subset = [captions[i] for i in perm[:n]]
        out_path = out_dir / f"{prefix}_N{n}.jsonl"
        write_jsonl(out_path, subset)
        print(f"[OK] wrote {n:>6d} prompts -> {out_path}")
if __name__ == "__main__":
    main()