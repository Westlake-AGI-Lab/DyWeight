import os
import argparse
from tqdm import tqdm
import ImageReward as RM
import json

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
    args = ap.parse_args()
    model = RM.load("path_to/ImageReward/ImageReward.pt")
    prompts = load_prompts(args.prompts_file)
    files = sorted([
        f for f in os.listdir(args.gen_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    assert len(files) > 0, f"No image files found in {args.gen_dir}"
    score_sum = 0.0
    count = 0
    for fn in tqdm(files, desc="imagereward"):
        # expected: 000123_01.png
        stem = os.path.splitext(fn)[0]
        prompt_idx = int(stem.split("_")[0])
        assert 0 <= prompt_idx < len(prompts), f"prompt_idx out of range: {prompt_idx}, file={fn}"
        prompt = prompts[prompt_idx]
        img_path = os.path.join(args.gen_dir, fn)
        score = float(model.score(prompt, img_path))
        score_sum += score
        count += 1
    mean_score = score_sum / count
    print(f"[ImageReward] path={args.gen_dir}")
    print(f"[ImageReward] count={count} mean={mean_score:.6f}")
if __name__ == "__main__":
    main()