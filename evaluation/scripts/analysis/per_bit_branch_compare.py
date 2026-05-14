"""Per-bit comparison of the spatial-only and spectral-only branches.

The dual_branch_r50 ablation showed (mean over the test set):

  full        : 99.66 % mean bit / 97.27 % exact
  no_spatial  : 99.61 / 96.88   (spectral alone)
  no_spectral : 59.47 /  1.95   (spatial alone)

The aggregate ``59.5 %'' isn't very informative -- with 8 independent bits
the spatial branch could be reading any subset of bits at high accuracy and
the rest at chance, or *all* bits at uniform 59 %. This script splits that
apart: per-bit accuracy in each ablation mode, plus a bit-by-bit confusion
between the two branches' predictions.

Concretely, for each bit i:
  spatial[i] : accuracy of (no_spectral) on bit i
  spectral[i]: accuracy of (no_spatial) on bit i
  agreement[i]: P(spatial-pred == spectral-pred), conditional on each bit value
  exclusivity[i]: P(spatial right ∧ spectral wrong) - the bits the spatial
                  branch *adds* on top of the spectral branch.

Verdict heuristics:
  - if spatial[i] tracks 0.5 for all i, the spatial branch is uniformly
    guessing and the +1.9 pp full-vs-spectral lift comes from reducing
    spectral's residual error rather than adding new bits;
  - if spatial[i] is bimodal (some bits ~ 0.95, others ~ 0.5), there are
    specific bits the spatial branch ``carries''.

Outputs:
  - evaluation/results/metrics/per_bit_branch_compare.json
  - evaluation/results/figures/per_bit_branch_compare.png

Usage:
  python evaluation/scripts/analysis/per_bit_branch_compare.py \
    --config decoding/configs/dual_branch_r50.yaml \
    --checkpoint decoding/checkpoints/dual_branch_r50.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DECODING_ROOT = REPO_ROOT / "decoding"
sys.path.insert(0, str(REPO_ROOT))

from decoding.data.dataset import WatermarkDataset
from decoding.models import get_model
from decoding.models.dual_branch import DualBranchDecoder


@torch.no_grad()
def predict(model: DualBranchDecoder, x: torch.Tensor, mode: str) -> torch.Tensor:
    spatial_feat = model.spatial(x)
    spectral_feat = model.spec_encoder(model._spectrum(x))
    if mode == "no_spectral":
        spectral_feat = torch.zeros_like(spectral_feat)
    elif mode == "no_spatial":
        spatial_feat = torch.zeros_like(spatial_feat)
    elif mode != "full":
        raise ValueError(mode)
    logits = model.fusion(torch.cat([spatial_feat, spectral_feat], dim=-1))
    return (torch.sigmoid(logits) > 0.5).float()


def gather(model, loader, device):
    out = {"full": [], "no_spectral": [], "no_spatial": []}
    targets = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["bits"]
        targets.append(y)
        for mode in out:
            out[mode].append(predict(model, x, mode).cpu())
    targets = torch.cat(targets).numpy().astype(np.int32)
    out = {k: torch.cat(v).numpy().astype(np.int32) for k, v in out.items()}
    return out, targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--config", default=str(DECODING_ROOT / "configs" / "dual_branch_r50.yaml"))
    p.add_argument("--checkpoint", default=str(DECODING_ROOT / "checkpoints" / "dual_branch_r50.pth"))
    p.add_argument("--arch", default=None)
    p.add_argument("--metadata", default=None)
    p.add_argument("--images", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-json", default=str(REPO_ROOT / "evaluation" / "results" / "metrics" / "per_bit_branch_compare.json"))
    p.add_argument("--out-fig", default=str(REPO_ROOT / "evaluation" / "results" / "figures" / "per_bit_branch_compare.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    metadata = args.metadata or os.path.join(DECODING_ROOT, cfg["data"]["metadata_path"])
    images = args.images or os.path.join(DECODING_ROOT, cfg["data"]["images_path"])
    splits = args.splits or os.path.join(DECODING_ROOT, cfg["data"]["splits_path"])
    image_size = args.image_size or cfg["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = args.arch or cfg["model"]["architecture"]

    model = get_model(arch, num_outputs=8, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full = WatermarkDataset(metadata_path=metadata, image_dir=images, transform=transform)
    with open(splits) as f:
        split_idx = json.load(f)
    test_set = Subset(full, split_idx["test"])
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"[per_bit_compare] n_test={len(test_set)}")

    preds, targets = gather(model, loader, device)
    full_p = preds["full"]
    spec_p = preds["no_spatial"]      # spectral alone
    spat_p = preds["no_spectral"]     # spatial alone

    n_bits = targets.shape[1]
    rows = []
    for i in range(n_bits):
        y = targets[:, i]
        full_acc = float((full_p[:, i] == y).mean())
        spec_acc = float((spec_p[:, i] == y).mean())
        spat_acc = float((spat_p[:, i] == y).mean())
        # bits where spatial is right and spectral is wrong on this bit
        spat_right = (spat_p[:, i] == y)
        spec_wrong = (spec_p[:, i] != y)
        spat_recovers = float((spat_right & spec_wrong).sum() / max((spec_wrong).sum(), 1))
        agreement = float((spat_p[:, i] == spec_p[:, i]).mean())
        rows.append({
            "bit": i,
            "full_acc": full_acc,
            "spectral_alone_acc": spec_acc,
            "spatial_alone_acc": spat_acc,
            "spatial_recovers_spectral_errors": spat_recovers,
            "branch_agreement": agreement,
        })
        print(
            f"  bit {i}: full={full_acc:.4f}  spec={spec_acc:.4f}  spat={spat_acc:.4f}  "
            f"spat-recovers-spec-errors={spat_recovers:.3f}  agree={agreement:.3f}"
        )

    out = {
        "checkpoint": args.checkpoint,
        "n_test": int(targets.shape[0]),
        "per_bit": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nresults: {out_json}")

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    bits = np.arange(n_bits)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(bits - 0.25, [r["full_acc"] for r in rows], width=0.25, label="full")
    ax.bar(bits, [r["spectral_alone_acc"] for r in rows], width=0.25, label="spectral alone")
    ax.bar(bits + 0.25, [r["spatial_alone_acc"] for r in rows], width=0.25, label="spatial alone")
    ax.axhline(0.5, color="grey", linestyle=":")
    ax.set_xticks(bits)
    ax.set_xlabel("bit")
    ax.set_ylabel("test accuracy")
    ax.set_title("Per-bit accuracy by branch (zeroing the other branch's features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)
    print(f"figure: {out_fig}")


if __name__ == "__main__":
    main()
