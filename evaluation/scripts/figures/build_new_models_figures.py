"""Build figures comparing the four new decoders against the old baselines.

Inputs:
  - Per-architecture training logs (one per arch) at <logs-dir>/<arch>.log,
    produced by ``train_new_decoders.sh``. Each log contains epoch-by-epoch
    "Train loss" / "Mean bit accuracy" / "Exact match rate" lines.
  - Per-architecture clean+robustness JSONs under <staging-root>/<arch>/.
  - Old-model results under <old-results-dir>/ (robustness_*.json,
    architecture_comparison.md, test_results/baseline_resnet50.json).

Outputs (under <figures-root>):
  - training_loss.png             train loss per epoch, all 4 new archs
  - training_val_curves.png       val mean-bit + exact-match per epoch, all 4
  - clean_comparison.png          bar chart, all 7 models on clean test set
  - per_bit_new_models.png        per-bit clean accuracy, 4 new models
  - robustness_heatmap.png        7 models x 6 attacks heatmap
  - dual_branch_vs_resnet.png     head-to-head per-bit + per-attack
  - figures_manifest.json         what got generated and from what

The matplotlib style follows ``create_figures.py``: filled bars with a thin
black edge, value labels on top, sensible y-limits, no fancy themes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NEW_ARCHS = ["global_stats", "spectral", "multiscale_pyramid", "dual_branch", "dual_branch_r50"]

# Map staging-dir name -> log filename (multiscale.yaml's experiment name is
# multiscale_pyramid but its log is multiscale.log).
LOG_NAME_BY_ARCH = {
    "global_stats": "global_stats.log",
    "spectral": "spectral.log",
    "multiscale_pyramid": "multiscale.log",
    "dual_branch": "dual_branch.log",
    "dual_branch_r50": "dual_branch_r50.log",
}

OLD_DISPLAY = {
    "resnet": "ResNet-50 (shared)",
    "separate": "8x ResNet-50 (separate)",
    "vit": "ViT-B/16",
}

NEW_DISPLAY = {
    "global_stats": "GlobalStats (MLP)",
    "spectral": "Spectral (FFT-CNN)",
    "multiscale_pyramid": "MultiScale Pyramid",
    "dual_branch": "DualBranch R-18 (512)",
    "dual_branch_r50": "DualBranch R-50 (1024)",
}

# Order used in every comparison plot (best-known clean accuracy descending;
# dual_branch_r50 gets pole position because it's the final architecture).
DISPLAY_ORDER = [
    ("dual_branch_r50", "new", NEW_DISPLAY["dual_branch_r50"]),
    ("dual_branch", "new", NEW_DISPLAY["dual_branch"]),
    ("resnet", "old", OLD_DISPLAY["resnet"]),
    ("separate", "old", OLD_DISPLAY["separate"]),
    ("multiscale_pyramid", "new", NEW_DISPLAY["multiscale_pyramid"]),
    ("spectral", "new", NEW_DISPLAY["spectral"]),
    ("global_stats", "new", NEW_DISPLAY["global_stats"]),
    ("vit", "old", OLD_DISPLAY["vit"]),
]


# ---------------------------- log parsing ---------------------------- #

EPOCH_RE = re.compile(r"^Epoch\s+(\d+)\s*/\s*(\d+)\s*$")
TRAIN_LOSS_RE = re.compile(r"^Train loss:\s*([0-9.]+)\s*$")
VAL_MEAN_RE = re.compile(r"Mean bit accuracy:\s*([0-9.]+)")
VAL_EXACT_RE = re.compile(r"Exact match rate:\s*([0-9.]+)")
BEST_EXACT_RE = re.compile(r"Best exact match rate:\s*([0-9.]+)")


def parse_training_log(path: Path) -> dict:
    """Extract per-epoch trajectory from a train.py log.

    Returns ``{"epochs": [...], "train_loss": [...], "val_mean": [...],
    "val_exact": [...], "best_val_exact": float|None}``. The first occurrence
    of "Mean bit accuracy" *after* a train-loss line is treated as the val
    metric (matches the order in ``train.py``: train -> val).
    """
    out = {"epochs": [], "train_loss": [], "val_mean": [], "val_exact": [],
           "best_val_exact": None}
    if not path.exists():
        return out

    cur_epoch = None
    cur_train = None
    cur_val_mean = None
    cur_val_exact = None
    seen_val_mean = False  # only the first metric block per epoch (val, not test)

    def commit():
        if cur_epoch is not None and cur_train is not None and cur_val_mean is not None:
            out["epochs"].append(cur_epoch)
            out["train_loss"].append(cur_train)
            out["val_mean"].append(cur_val_mean)
            out["val_exact"].append(cur_val_exact if cur_val_exact is not None else float("nan"))

    with path.open("r", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            m = EPOCH_RE.match(line)
            if m:
                commit()
                cur_epoch = int(m.group(1))
                cur_train = None
                cur_val_mean = None
                cur_val_exact = None
                seen_val_mean = False
                continue
            m = TRAIN_LOSS_RE.match(line)
            if m and cur_train is None:
                cur_train = float(m.group(1))
                continue
            m = VAL_MEAN_RE.search(line)
            if m and cur_train is not None and not seen_val_mean:
                cur_val_mean = float(m.group(1))
                seen_val_mean = True
                continue
            m = VAL_EXACT_RE.search(line)
            if m and seen_val_mean and cur_val_exact is None:
                cur_val_exact = float(m.group(1))
                continue
            m = BEST_EXACT_RE.search(line)
            if m:
                out["best_val_exact"] = float(m.group(1))
        commit()
    return out


# ---------------------------- result loaders ---------------------------- #

def load_clean_metrics(staging_root: Path, arch: str) -> Optional[dict]:
    p = staging_root / arch / "clean_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_robustness(staging_root: Path, arch: str) -> Optional[dict]:
    p = staging_root / arch / "robustness.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_old_robustness(old_results_dir: Path, name: str) -> Optional[dict]:
    p = old_results_dir / f"robustness_{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_old_clean_from_md(old_results_dir: Path) -> dict[str, dict]:
    """Parse the architecture_comparison.md table for old-model clean metrics.

    The table has one row per arch with bit columns + mean + exact. We only
    need mean + exact + per-bit for the comparison plots.
    """
    md = old_results_dir / "architecture_comparison.md"
    if not md.exists():
        return {}
    out: dict[str, dict] = {}
    name_to_key = {
        "ResNet-50 (shared backbone)": "resnet",
        "8x ResNet-50 (separate)": "separate",
        "ViT-B/16": "vit",
    }
    for line in md.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 11:
            continue
        # Skip header / separator rows
        if cells[0] in ("Architecture",) or set(cells[0]) <= {"-"}:
            continue
        if cells[0] not in name_to_key:
            continue
        try:
            per_bit = [float(x) for x in cells[1:9]]
            mean = float(cells[9])
            exact = float(cells[10])
        except ValueError:
            continue
        out[name_to_key[cells[0]]] = {
            "per_bit_accuracy": per_bit,
            "mean_bit_accuracy": mean,
            "exact_match_rate": exact,
        }
    return out


# ---------------------------- plotters ---------------------------- #

PALETTE = {
    "global_stats": "#94a3b8",
    "spectral": "#64748b",
    "multiscale_pyramid": "#0891b2",
    "dual_branch": "#16a34a",
    "dual_branch_r50": "#15803d",
    "resnet": "#2563eb",
    "separate": "#7c3aed",
    "vit": "#ef4444",
}


def plot_training_loss(traj: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for arch in NEW_ARCHS:
        t = traj.get(arch, {})
        if not t.get("epochs"):
            continue
        ax.plot(t["epochs"], t["train_loss"], marker="o", linewidth=1.5,
                markersize=4, color=PALETTE[arch], label=NEW_DISPLAY[arch])
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train loss (BCE)", fontsize=12)
    ax.set_title("Training loss per epoch (new decoders)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_training_val_curves(traj: dict[str, dict], out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for arch in NEW_ARCHS:
        t = traj.get(arch, {})
        if not t.get("epochs"):
            continue
        c = PALETTE[arch]
        ax1.plot(t["epochs"], t["val_mean"], marker="o", linewidth=1.5,
                 markersize=4, color=c, label=NEW_DISPLAY[arch])
        ax2.plot(t["epochs"], t["val_exact"], marker="o", linewidth=1.5,
                 markersize=4, color=c, label=NEW_DISPLAY[arch])

    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="chance")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val mean bit accuracy")
    ax1.set_title("Val mean bit accuracy per epoch", fontweight="bold")
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val exact match rate")
    ax2.set_title("Val exact match per epoch", fontweight="bold")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_clean_comparison(
    new_clean: dict[str, dict],
    old_clean: dict[str, dict],
    out_path: Path,
) -> None:
    items = []
    for key, kind, label in DISPLAY_ORDER:
        if kind == "new":
            data = new_clean.get(key)
            if data is not None:
                tm = data.get("test_metrics") or data
                items.append((label, key, tm.get("mean_bit_accuracy"), tm.get("exact_match_rate")))
        else:
            data = old_clean.get(key)
            if data is not None:
                items.append((label, key, data.get("mean_bit_accuracy"), data.get("exact_match_rate")))
    if not items:
        return

    labels = [it[0] for it in items]
    keys = [it[1] for it in items]
    means = [it[2] for it in items]
    exacts = [it[3] for it in items]
    colors = [PALETTE[k] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(labels))

    bars1 = ax1.bar(x, [m * 100 for m in means], color=colors, edgecolor="black", linewidth=0.7)
    ax1.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="chance (50%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Mean bit accuracy (%)", fontsize=11)
    ax1.set_ylim(40, 100)
    ax1.set_title("Clean test set: mean bit accuracy", fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)

    bars2 = ax2.bar(x, [e * 100 for e in exacts], color=colors, edgecolor="black", linewidth=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Exact match rate (%)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.set_title("Clean test set: exact match (8/8 bits)", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars2, exacts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                 f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_bit_new(new_clean: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    bits = np.arange(8)
    width = 0.2
    for j, arch in enumerate(NEW_ARCHS):
        data = new_clean.get(arch)
        if data is None:
            continue
        per_bit = (data.get("test_metrics") or data).get("per_bit_accuracy")
        if not per_bit:
            continue
        ax.bar(bits + j * width, [v * 100 for v in per_bit], width=width,
               color=PALETTE[arch], edgecolor="black", linewidth=0.5,
               label=NEW_DISPLAY[arch])
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="chance")
    ax.set_xticks(bits + 1.5 * width)
    ax.set_xticklabels([f"bit {i}" for i in range(8)])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 100)
    ax.set_title("Per-bit clean accuracy: new decoders", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_robustness_heatmap(
    new_rob: dict[str, dict],
    old_rob: dict[str, dict],
    out_path: Path,
) -> None:
    attack_order = ["clean", "jpeg_q90", "jpeg_q75", "jpeg_q50", "resize_512", "random_crop_75"]

    rows = []
    row_labels = []
    for key, kind, label in DISPLAY_ORDER:
        src = new_rob.get(key) if kind == "new" else old_rob.get(key)
        if src is None:
            continue
        results = src.get("results", {})
        row = []
        for atk in attack_order:
            v = results.get(atk, {}).get("mean_bit_accuracy")
            row.append(float(v) if v is not None else np.nan)
        rows.append(row)
        row_labels.append(label)
    if not rows:
        return

    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 2.0))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(attack_order)))
    ax.set_xticklabels(attack_order, rotation=15, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=10)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                txt = "n/a"
                color = "black"
            else:
                txt = f"{v:.3f}"
                color = "black" if 0.55 < v < 0.95 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    ax.set_title("Robustness: mean bit accuracy under attack", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mean bit accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dual_branch_vs_resnet(
    new_clean: dict[str, dict],
    old_clean: dict[str, dict],
    new_rob: dict[str, dict],
    old_rob: dict[str, dict],
    out_path: Path,
) -> None:
    db_clean = new_clean.get("dual_branch")
    rn_clean = old_clean.get("resnet")
    db_rob = new_rob.get("dual_branch", {}).get("results", {})
    rn_rob = old_rob.get("resnet", {}).get("results", {})

    if not (db_clean and rn_clean and db_rob and rn_rob):
        return

    db_per_bit = (db_clean.get("test_metrics") or db_clean).get("per_bit_accuracy", [])
    rn_per_bit = rn_clean.get("per_bit_accuracy", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    bits = np.arange(8)
    w = 0.35
    ax1.bar(bits - w / 2, [v * 100 for v in rn_per_bit], width=w, color=PALETTE["resnet"],
            edgecolor="black", linewidth=0.5, label=OLD_DISPLAY["resnet"])
    ax1.bar(bits + w / 2, [v * 100 for v in db_per_bit], width=w, color=PALETTE["dual_branch"],
            edgecolor="black", linewidth=0.5, label=NEW_DISPLAY["dual_branch"])
    ax1.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_xticks(bits)
    ax1.set_xticklabels([f"bit {i}" for i in range(8)])
    ax1.set_ylim(40, 105)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Per-bit clean accuracy", fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="lower right", fontsize=10)

    attacks = ["clean", "jpeg_q90", "jpeg_q75", "jpeg_q50", "resize_512", "random_crop_75"]
    rn_vals = [rn_rob.get(a, {}).get("mean_bit_accuracy", np.nan) for a in attacks]
    db_vals = [db_rob.get(a, {}).get("mean_bit_accuracy", np.nan) for a in attacks]
    x = np.arange(len(attacks))
    ax2.bar(x - w / 2, [v * 100 for v in rn_vals], width=w, color=PALETTE["resnet"],
            edgecolor="black", linewidth=0.5, label=OLD_DISPLAY["resnet"])
    ax2.bar(x + w / 2, [v * 100 for v in db_vals], width=w, color=PALETTE["dual_branch"],
            edgecolor="black", linewidth=0.5, label=NEW_DISPLAY["dual_branch"])
    ax2.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(attacks, rotation=15, ha="right")
    ax2.set_ylim(40, 105)
    ax2.set_ylabel("Mean bit accuracy (%)")
    ax2.set_title("Robustness across attacks", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="lower right", fontsize=10)

    fig.suptitle("DualBranch (new) vs ResNet-50 (old)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------- main ---------------------------- #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--staging-root", required=True,
                   help="Per-arch staging dir (e.g. /workspace/new_models)")
    p.add_argument("--figures-root", required=True,
                   help="Where to write figures (e.g. /workspace/new_models_figures)")
    p.add_argument("--logs-dir", required=True,
                   help="Training-log dir (e.g. decoding/.train_new/logs)")
    p.add_argument("--old-results-dir", required=True,
                   help="Old-model results dir (e.g. decoding/results)")
    args = p.parse_args()

    staging_root = Path(args.staging_root)
    figures_root = Path(args.figures_root)
    logs_dir = Path(args.logs_dir)
    old_results_dir = Path(args.old_results_dir)

    figures_root.mkdir(parents=True, exist_ok=True)

    # 1. parse logs
    traj: dict[str, dict] = {}
    for arch in NEW_ARCHS:
        log_name = LOG_NAME_BY_ARCH.get(arch, f"{arch}.log")
        traj[arch] = parse_training_log(logs_dir / log_name)

    # 2. load eval JSONs
    new_clean = {a: load_clean_metrics(staging_root, a) for a in NEW_ARCHS}
    new_rob = {a: load_robustness(staging_root, a) for a in NEW_ARCHS}
    old_rob = {n: load_old_robustness(old_results_dir, n) for n in OLD_DISPLAY}
    old_clean = load_old_clean_from_md(old_results_dir)

    # 3. plots
    out = figures_root
    plot_training_loss(traj, out / "training_loss.png")
    plot_training_val_curves(traj, out / "training_val_curves.png")
    plot_clean_comparison(new_clean, old_clean, out / "clean_comparison.png")
    plot_per_bit_new(new_clean, out / "per_bit_new_models.png")
    plot_robustness_heatmap(new_rob, old_rob, out / "robustness_heatmap.png")
    plot_dual_branch_vs_resnet(new_clean, old_clean, new_rob, old_rob,
                               out / "dual_branch_vs_resnet.png")

    # 4. manifest
    manifest = {
        "trajectories": {
            arch: {
                "n_epochs": len(t["epochs"]),
                "final_train_loss": t["train_loss"][-1] if t["train_loss"] else None,
                "final_val_mean": t["val_mean"][-1] if t["val_mean"] else None,
                "final_val_exact": t["val_exact"][-1] if t["val_exact"] else None,
                "best_val_exact": t["best_val_exact"],
            }
            for arch, t in traj.items()
        },
        "have_clean": {a: new_clean[a] is not None for a in NEW_ARCHS},
        "have_robustness": {a: new_rob[a] is not None for a in NEW_ARCHS},
        "have_old_clean": list(old_clean.keys()),
        "have_old_robustness": [n for n, v in old_rob.items() if v is not None],
        "figures": sorted(p.name for p in figures_root.glob("*.png")),
    }
    (figures_root / "figures_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("[figures] wrote:")
    for png in sorted(figures_root.glob("*.png")):
        print(f"  {png}")
    print(f"[figures] manifest: {figures_root / 'figures_manifest.json'}")


if __name__ == "__main__":
    main()
