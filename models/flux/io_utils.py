import torch
import imageio
import logging
import os, json, glob, re, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
from PIL import Image
from time import time, strftime
from typing import List, Optional, Dict
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def create_logger(exp_dir, accelerator):
    """
    Create a logger that writes to a log file and stdout.
    """
    if accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{exp_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def print_with_prefix(*messages):
    prefix = f"\033[34m[DyWeight {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}", flush=True)
    
def read_prompts(path: str) -> List[str]:
    assert path and os.path.isfile(path)
    if path.endswith(".jsonl"):
        out: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                out.append(json.loads(ln)["prompt"])
        return out
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return df["caption"].astype(str).tolist()
    raise ValueError("prompts_path must be .jsonl or .txt")
    
# visualization
def make_png2gif(frames_dir: str, out_gif: str = None, pattern: str = "step*.png", fps: int = 8) -> str | None:
    frames_dir = str(frames_dir)
    paths = sorted(
        glob.glob(os.path.join(frames_dir, pattern)),
        key=lambda p: int(re.findall(r"step(\d+)", os.path.basename(p))[0])
        if re.findall(r"step(\d+)", os.path.basename(p)) else 0
    )
    imgs = [imageio.v2.imread(p) for p in paths]
    imageio.mimsave(out_gif, imgs, duration=1.0 / fps)
    return out_gif
def visual_teacher_student(teacher_pils, student_pils, out_png=None):
    pad, col_gap, row_gap = 16, 16, 12
    idxs = [0] if len(teacher_pils) == 1 else [0, len(teacher_pils) - 1]
    rows = len(idxs)
    w, h = teacher_pils[0].size
    W = pad + w + col_gap + w + pad
    H = pad + rows * h + (rows - 1) * row_gap + pad
    canvas = Image.new("RGB", (W, H), color=(255, 255, 255))
    for r, i in enumerate(idxs):
        top = pad + r * (h + row_gap)
        canvas.paste(teacher_pils[i], (pad, top))
        canvas.paste(student_pils[i], (pad + w + col_gap, top))
    if out_png is not None:
        canvas.save(out_png)
    else:
        canvas.save("./teacher_student_visualize.png")
# ------------loss-------------
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.figsize': (3.4, 2.4),
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    "mathtext.fontset": "stix",
})
def plot_loss_fig(run_dir, log_loss_list, beta=0.98, use_log_scale=False):
    csv_path = os.path.join(run_dir, "loss_train.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        w.writerows([[i, float(v)] for i, v in enumerate(log_loss_list, 1)])
    loss = np.asarray(log_loss_list, dtype=float)
    x = np.arange(1, len(loss) + 1)
    ema = np.zeros_like(loss)
    mu = 0
    for i in range(len(loss)):
        mu = beta * mu + (1 - beta) * loss[i]
        ema[i] = mu / (1 - beta**(i + 1))
    plt.figure(figsize=(10, 4), dpi=300)
    raw_color = "#E91E63"
    ema_color = "#8E0038"
    plt.plot(x, loss, color=raw_color, linewidth=0.3, alpha=0.15, label="Raw")
    plt.plot(x, ema,  color=ema_color, linewidth=1.2, alpha=0.9, label=f"EMA")
    if use_log_scale:
        plt.yscale('log')
        plt.ylabel("Loss (Log)")
    else:
        plt.ylabel("Loss")
    plt.xlabel("Steps")
    plt.title("Training Loss") 
    plt.legend(loc='upper right', frameon=False) 
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "loss_train.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_weights_trend(
    dyweight_params,
    out_png: str,
    vmin: float = -5.0,
    vmax: float = 5.0,
    cell_px: int = 100,
    dpi: int = 200,
    font_min: int = 1,
    font_max: int = 4,
):
    K = int(dyweight_params.K)
    W = torch.zeros(K, K, dtype=torch.float32)
    for i in range(K):
        wr = dyweight_params.weights_row(i)
        if isinstance(wr, tuple):
            row, start = wr
        else:
            row = wr
            start = i + 1 - row.shape[0]
        W[i, start:i+1] = row.detach().float().cpu()
    W = W.numpy()
    if hasattr(dyweight_params, "win_mask") and dyweight_params.win_mask is not None:
        mask = (~dyweight_params.win_mask.detach().cpu().numpy())
    else:
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)

    Wm = np.ma.array(W, mask=mask)
    _CMAPlocal = LinearSegmentedColormap.from_list(
        "bwo", [(0.121, 0.466, 0.705), (1.0, 1.0, 1.0), (1.0, 0.498, 0.055)], N=256,)
    _CMAPlocal.set_bad(color="white")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    fig_w = (K * cell_px) / dpi
    fig_h = (K * cell_px) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(Wm, cmap=_CMAPlocal, norm=norm, origin="upper", interpolation="nearest")
    ax.set_axis_off()
    for s in ax.spines.values(): s.set_visible(False)

    fs = int(min(font_max, max(font_min, cell_px * 0.36)))
    yy, xx = np.where(~mask)
    for y, x in zip(yy, xx):
        ax.text(x, y, f"{W[y, x]:.5f}", ha="center", va="center", fontsize=fs, color="black", clip_on=False)

    fig.patch.set_facecolor("white")
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)