"""Ablate DualBranchDecoder's spatial / spectral branches and report accuracy.

The fusion head takes ``concat([spatial_feat, spectral_feat])`` (see
``DualBranchDecoder.forward``). At eval time we can selectively zero one
branch's features before fusion to measure how much each branch actually
contributes:

  full          - normal forward (both branches active)
  no_spectral   - spectral_feat replaced with zeros (spatial-only signal)
  no_spatial    - spatial_feat replaced with zeros (spectral-only signal)

Outputs per-bit accuracy, mean bit accuracy, and exact match for each mode,
plus the deltas vs. the full model. If no_spectral matches full to within a
fraction of a percent, the spectral branch is along for the ride and could
be dropped; if no_spatial collapses to ~0.5, the spatial branch is doing all
the work.

Usage:
  python decoding/scripts/ablate_dual_branch.py \\
    --config decoding/configs/dual_branch.yaml \\
    --checkpoint decoding/checkpoints/dual_branch.pth

Config-driven by default; CLI overrides for data paths if you need them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent
sys.path.insert(0, str(DECODING_ROOT))

from src.dataloader import WatermarkDataset
from src.models import get_model
from src.models.dual_branch import DualBranchDecoder
from src.utils import compute_metrics


MODES = ["full", "no_spectral", "no_spatial"]


@torch.no_grad()
def forward_with_mode(model: DualBranchDecoder, x: torch.Tensor, mode: str) -> torch.Tensor:
    """Mirror DualBranchDecoder.forward but optionally zero one branch.

    Keeps the same arithmetic the network was trained with: both branches
    are always *evaluated* (so the spectrum normalization and spatial pool
    statistics are unchanged); only the feature vector handed to fusion is
    zeroed.
    """
    spatial_feat = model.spatial(x)
    spectral_feat = model.spec_encoder(model._spectrum(x))
    if mode == "no_spectral":
        spectral_feat = torch.zeros_like(spectral_feat)
    elif mode == "no_spatial":
        spatial_feat = torch.zeros_like(spatial_feat)
    elif mode != "full":
        raise ValueError(f"unknown mode: {mode}")
    return model.fusion(torch.cat([spatial_feat, spectral_feat], dim=-1))


def evaluate_mode(model, loader, device, mode: str) -> dict:
    all_preds, all_targets = [], []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["bits"]
        logits = forward_with_mode(model, images, mode)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        all_preds.append(preds)
        all_targets.append(targets)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    return {
        "per_bit_accuracy": [float(x) for x in metrics["per_bit_accuracy"]],
        "mean_bit_accuracy": float(metrics["mean_bit_accuracy"]),
        "exact_match_rate": float(metrics["exact_match_rate"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(DECODING_ROOT / "configs" / "dual_branch.yaml"),
                   help="Training config (used for image_size + default data paths).")
    p.add_argument("--arch", type=str, default=None,
                   help="Override architecture name (default: read from config). "
                        "Use this to ablate dual_branch_r50, dual_branch_r34, etc.")
    p.add_argument("--checkpoint", type=str, default=str(DECODING_ROOT / "checkpoints" / "dual_branch.pth"))
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--images", type=str, default=None)
    p.add_argument("--splits", type=str, default=None)
    p.add_argument("--image-size", type=int, default=None,
                   help="Override config image_size.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-json", type=str,
                   default=str(DECODING_ROOT / "results" / "dual_branch_ablation.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Config paths are relative to decoding/.
    cfg_dir = DECODING_ROOT
    metadata = args.metadata or os.path.join(cfg_dir, cfg["data"]["metadata_path"])
    images = args.images or os.path.join(cfg_dir, cfg["data"]["images_path"])
    splits = args.splits or os.path.join(cfg_dir, cfg["data"]["splits_path"])
    image_size = args.image_size or cfg["data"]["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ablate_dual_branch ===")
    print(f"  device:      {device}")
    print(f"  checkpoint:  {args.checkpoint}")
    print(f"  config:      {args.config}")
    print(f"  image_size:  {image_size}")

    # Load model. ``--arch`` overrides the config's architecture name so the
    # same script can ablate dual_branch_r50 / dual_branch_r34 checkpoints.
    arch = args.arch or cfg["model"]["architecture"]
    print(f"  arch:        {arch}")
    model = get_model(arch, num_outputs=8, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    val_exact = ckpt.get("metrics", {}).get("exact_match_rate") if isinstance(ckpt, dict) else None
    print(f"  ckpt epoch:  {epoch}")
    if val_exact is not None:
        print(f"  ckpt val_exact_match: {val_exact:.4f}")

    # Build test loader (matches evaluate.py).
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
    print(f"  test images: {len(test_set)}")

    # Run all three modes.
    results = {}
    for mode in MODES:
        print(f"\n--- evaluating mode: {mode} ---")
        results[mode] = evaluate_mode(model, loader, device, mode)
        m = results[mode]
        print(f"  mean_bit={m['mean_bit_accuracy']:.4f}  exact={m['exact_match_rate']:.4f}")
        print(f"  per_bit={[f'{x:.3f}' for x in m['per_bit_accuracy']]}")

    # Deltas vs. full.
    full_m = results["full"]
    deltas = {}
    for mode in MODES:
        if mode == "full":
            continue
        m = results[mode]
        deltas[mode] = {
            "delta_mean_bit_accuracy": m["mean_bit_accuracy"] - full_m["mean_bit_accuracy"],
            "delta_exact_match_rate": m["exact_match_rate"] - full_m["exact_match_rate"],
            "delta_per_bit_accuracy": [
                a - b for a, b in zip(m["per_bit_accuracy"], full_m["per_bit_accuracy"])
            ],
        }

    # Print summary table.
    print()
    print("=" * 78)
    print("Summary (ablation of DualBranchDecoder branches on clean test set)")
    print("=" * 78)
    header = f"  {'mode':<14}  {'mean_bit':>9}  {'exact':>7}  " + "  ".join(f"b{i}" for i in range(8))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode in MODES:
        m = results[mode]
        bits = "  ".join(f"{x:.2f}" for x in m["per_bit_accuracy"])
        print(f"  {mode:<14}  {m['mean_bit_accuracy']:>9.4f}  {m['exact_match_rate']:>7.4f}  {bits}")

    print()
    print("Deltas vs. full:")
    for mode, d in deltas.items():
        print(f"  {mode:<14}  Δmean={d['delta_mean_bit_accuracy']:+.4f}  "
              f"Δexact={d['delta_exact_match_rate']:+.4f}")
        per_bit = "  ".join(f"{x:+.2f}" for x in d["delta_per_bit_accuracy"])
        print(f"                 per-bit Δ: {per_bit}")

    # Verdict heuristic.
    no_spec = results["no_spectral"]["mean_bit_accuracy"]
    no_spat = results["no_spatial"]["mean_bit_accuracy"]
    full_mean = full_m["mean_bit_accuracy"]
    print()
    print("Verdict:")
    if abs(full_mean - no_spec) < 0.01:
        print("  * Spectral branch contributes <1% mean bit accuracy. Could be dropped.")
    else:
        print(f"  * Spectral branch contributes {(full_mean - no_spec) * 100:.2f} pp mean bit accuracy.")
    if no_spat < 0.55:
        print(f"  * Spatial-only zeroed → {no_spat:.4f}: spectral branch alone can't decode.")
    else:
        print(f"  * Spectral branch alone reaches {no_spat:.4f} (above chance).")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "ckpt_epoch": epoch,
        "ckpt_val_exact_match": val_exact,
        "image_size": image_size,
        "n_test": len(test_set),
        "modes": results,
        "deltas_vs_full": deltas,
    }, indent=2))
    print(f"\nresults: {out_path}")


if __name__ == "__main__":
    main()
