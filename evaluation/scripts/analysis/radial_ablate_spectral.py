"""Radial-band ablation of the spectral branch's input.

The puzzle is mechanistic: what feature of log |F(g)| does the spectral CNN
read to decode 8 bits at 99.6% accuracy? This script answers it by ablating
the spectral input and watching the downstream accuracy.

For each of three masking modes:
  - keep-inside-r:  zero everything outside an fft-shifted disk of radius r
  - keep-outside-r: zero everything inside  an fft-shifted disk of radius r
  - radial-band:    zero everything outside an annulus [r0, r1]

we sweep r over [1, 4, 8, 16, 32, 48, 64, 96, 128] and report mean-bit and
exact-match accuracy. Two important wrinkles:

  1. The model standardises the spectrum per-image *before* the CNN. Zeroing
     bins, then standardising, biases the result. We standardise *first* and
     then zero, by splicing into ``DualBranchDecoder._spectrum``: see
     ``forward_with_radial_mask``.

  2. The model uses ``log1p(|F(g)|)``. We zero post-log: the CNN sees
     ``standardised_log_mag * mask`` rather than the FFT itself.

Verdict heuristic:
  - if accuracy is flat in r (e.g., keep-only-r=128 ≈ full),  the bits are
    encoded in a tiny number of low-freq bins and the rest of the spectrum
    is decorative;
  - if accuracy is flat for keep-outside-r at r ≤ K (i.e., zeroing the
    central K^2 bins doesn't hurt), the bits are encoded in mid/high freqs;
  - if accuracy drops uniformly with both modes, the model uses a globally-
    distributed envelope feature -- consistent with a ``read the radial
    profile of |F(g)|'' story rather than ``read narrow Fourier peaks''.

Outputs:
  - evaluation/results/metrics/radial_ablation.json
  - evaluation/results/figures/radial_ablation.png

Usage:
  python evaluation/scripts/analysis/radial_ablate_spectral.py \
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
from decoding.common.metrics import compute_metrics


def _radial_mask(size: int, mode: str, r0: int, r1: int | None = None,
                 device=None) -> torch.Tensor:
    """Build a (1, 1, size, size) mask. fftshifted disk geometry."""
    cy = cx = size // 2
    yy, xx = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    rr = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    if mode == "keep_inside":
        m = (rr <= r0).float()
    elif mode == "keep_outside":
        m = (rr > r0).float()
    elif mode == "annulus":
        assert r1 is not None
        m = ((rr >= r0) & (rr <= r1)).float()
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return m.view(1, 1, size, size).to(device) if device is not None else m.view(1, 1, size, size)


@torch.no_grad()
def forward_with_radial_mask(
    model: DualBranchDecoder, x: torch.Tensor, mask: torch.Tensor | None,
    spatial_mode: str = "active",
) -> torch.Tensor:
    """Mirror DualBranchDecoder.forward but multiply the (already
    standardised) spectral input by ``mask`` before feeding it to the
    spectral encoder. ``spatial_mode`` lets the caller also zero the spatial
    branch (to isolate the spectral branch -- the ablation showed that's the
    one that matters)."""
    spatial_feat = model.spatial(x)
    if spatial_mode == "zeroed":
        spatial_feat = torch.zeros_like(spatial_feat)
    spec = model._spectrum(x)        # (B, 3, fft_size, fft_size), standardised
    if mask is not None:
        spec = spec * mask
    spectral_feat = model.spec_encoder(spec)
    return model.fusion(torch.cat([spatial_feat, spectral_feat], dim=-1))


def evaluate(model, loader, device, mask, spatial_mode) -> dict:
    all_preds, all_targets = [], []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["bits"]
        logits = forward_with_radial_mask(model, images, mask, spatial_mode=spatial_mode)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        all_preds.append(preds)
        all_targets.append(targets)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    m = compute_metrics(preds, targets)
    return {
        "per_bit_accuracy": [float(x) for x in m["per_bit_accuracy"]],
        "mean_bit_accuracy": float(m["mean_bit_accuracy"]),
        "exact_match_rate": float(m["exact_match_rate"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--config", default=str(DECODING_ROOT / "configs" / "dual_branch_r50.yaml"))
    p.add_argument("--checkpoint", default=str(DECODING_ROOT / "checkpoints" / "dual_branch_r50.pth"))
    p.add_argument("--arch", default=None, help="override arch name")
    p.add_argument("--metadata", default=None)
    p.add_argument("--images", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--spatial-mode", choices=["active", "zeroed"], default="zeroed",
                   help="In the dual_branch_r50 ablation, the spatial branch alone "
                        "only reaches 59 percent, so by default we zero it to isolate "
                        "the spectral branch's behaviour.")
    p.add_argument("--radii", default="1,4,8,16,32,48,64,96,128",
                   help="Comma-separated radii to sweep.")
    p.add_argument("--out-json", default=str(REPO_ROOT / "evaluation" / "results" / "metrics" / "radial_ablation.json"))
    p.add_argument("--out-fig", default=str(REPO_ROOT / "evaluation" / "results" / "figures" / "radial_ablation.png"))
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
    print(f"[radial_ablate] device={device} arch={arch} ckpt={args.checkpoint}")

    model = get_model(arch, num_outputs=8, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    if not isinstance(model, DualBranchDecoder):
        raise SystemExit(f"radial ablation requires DualBranchDecoder, got {type(model).__name__}")

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

    fft_size = model.fft_size
    radii = [int(r) for r in args.radii.split(",")]

    results: dict = {"fft_size": fft_size, "spatial_mode": args.spatial_mode}

    # Baseline: full spectrum, mask=None.
    print(f"\n--- baseline (no mask, spatial={args.spatial_mode}) ---")
    base = evaluate(model, loader, device, None, args.spatial_mode)
    print(f"  mean_bit={base['mean_bit_accuracy']:.4f}  exact={base['exact_match_rate']:.4f}")
    results["baseline"] = base

    # Sweep three modes.
    for mode in ("keep_inside", "keep_outside"):
        results[mode] = []
        for r in radii:
            mask = _radial_mask(fft_size, mode, r0=r, device=device)
            row = evaluate(model, loader, device, mask, args.spatial_mode)
            row["r"] = r
            row["frac_kept"] = float(mask.mean().item())
            print(f"  {mode:<13} r={r:>3}  kept={row['frac_kept']:.3f}  "
                  f"mean_bit={row['mean_bit_accuracy']:.4f}  exact={row['exact_match_rate']:.4f}")
            results[mode].append(row)

    # Annular sweep at fixed bandwidth.
    annuli = [(0, 8), (8, 16), (16, 32), (32, 48), (48, 64), (64, 96), (96, 128)]
    results["annulus"] = []
    for r0, r1 in annuli:
        mask = _radial_mask(fft_size, "annulus", r0=r0, r1=r1, device=device)
        row = evaluate(model, loader, device, mask, args.spatial_mode)
        row["r0"] = r0
        row["r1"] = r1
        row["frac_kept"] = float(mask.mean().item())
        print(f"  annulus     [{r0:>3},{r1:>3}]  kept={row['frac_kept']:.3f}  "
              f"mean_bit={row['mean_bit_accuracy']:.4f}  exact={row['exact_match_rate']:.4f}")
        results["annulus"].append(row)

    # Save + plot.
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nresults: {out_json}")

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rs = [row["r"] for row in results["keep_inside"]]
    inside = [row["mean_bit_accuracy"] for row in results["keep_inside"]]
    outside = [row["mean_bit_accuracy"] for row in results["keep_outside"]]
    axes[0].plot(rs, inside, "o-", label="keep_inside (zero outside r)")
    axes[0].plot(rs, outside, "o-", label="keep_outside (zero inside r)")
    axes[0].axhline(base["mean_bit_accuracy"], color="k", linestyle="--", label="full spectrum")
    axes[0].axhline(0.5, color="grey", linestyle=":", label="chance")
    axes[0].set_xlabel("radius r (FFT bins)")
    axes[0].set_ylabel("mean bit accuracy")
    axes[0].set_title("Disk-mask sweep")
    axes[0].legend()
    centres = [0.5 * (a["r0"] + a["r1"]) for a in results["annulus"]]
    accs = [a["mean_bit_accuracy"] for a in results["annulus"]]
    axes[1].plot(centres, accs, "o-")
    axes[1].axhline(base["mean_bit_accuracy"], color="k", linestyle="--", label="full spectrum")
    axes[1].axhline(0.5, color="grey", linestyle=":", label="chance")
    axes[1].set_xlabel("annulus centre r (FFT bins)")
    axes[1].set_ylabel("mean bit accuracy")
    axes[1].set_title("Annular keep-band sweep")
    axes[1].legend()
    fig.suptitle(f"Spectral-branch radial ablation (spatial branch {args.spatial_mode})")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)
    print(f"figure:  {out_fig}")


if __name__ == "__main__":
    main()
