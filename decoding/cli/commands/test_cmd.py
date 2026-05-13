"""`decoding.cli test` — evaluate one or more decoders on the test split.

Reuses src.dataloader.WatermarkDataset, src.models.get_model, and
src.utils.compute_metrics so the numbers are directly comparable to
scripts/evaluate.py and scripts/ablate_dual_branch.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import DECODING_ROOT, CHECKPOINTS_ROOT, MODEL_BUNDLES_ROOT
from ..checkpoints import (
    ResolvedCheckpoint,
    parse_overrides,
    resolve_checkpoint,
)
from ..devices import pick_device
from ..registry import ModelSpec, all_model_names, parse_models_arg


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", default="all",
                   help="`all` or comma-separated list. Aliases ok (baseline, "
                        "dual_branch_r50, spectral, ...). "
                        f"Known: {', '.join(all_model_names())}")
    p.add_argument("--model", dest="models", help="alias for --models")
    p.add_argument("--weights-root", default=str(MODEL_BUNDLES_ROOT),
                   help="Root directory containing checkpoints. Supports both "
                        "the model_bundles layout (<root>/<arch>/<arch>.pth) and "
                        "the flat layout (<root>/<arch>.pth). "
                        f"Default: $PROJECT_DATA_ROOT/decoding/model_bundles")
    p.add_argument("--checkpoint", action="append", default=None,
                   metavar="NAME=PATH",
                   help="Per-model checkpoint override; repeat to set several.")
    p.add_argument("--metadata", default=None,
                   help="Path to metadata.json. Defaults to "
                        "watermark_encoding/data/metadata.json with a fallback "
                        "to encoding/data/metadata.json.")
    p.add_argument("--images", default=None,
                   help="Path to the watermarked images directory.")
    p.add_argument("--splits", default=str(DECODING_ROOT / "data" / "splits.json"))
    p.add_argument("--split", default="test", choices=("test", "val", "train"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=None,
                   help="Override the per-model default image size.")
    p.add_argument("--device", default="auto")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of evaluated images (debug).")
    # Dual-branch weighting --------------------------------------------------
    p.add_argument("--dual-spatial-weight", type=float, default=None,
                   help="Spatial branch weight for dual_branch* models.")
    p.add_argument("--dual-spectral-weight", type=float, default=None,
                   help="Spectral branch weight for dual_branch* models.")
    p.add_argument("--dual-weights", default=None,
                   help="Compact form, e.g. spatial=0.0,spectral=1.0 .")
    # Output -----------------------------------------------------------------
    p.add_argument("--output", required=True,
                   help="Directory to write per-model JSON + summary.{json,md}")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would run (models, weights, configs) and exit.")


def _parse_dual_weights(args: argparse.Namespace) -> Tuple[float, float]:
    spatial: Optional[float] = args.dual_spatial_weight
    spectral: Optional[float] = args.dual_spectral_weight
    if args.dual_weights:
        for raw in args.dual_weights.split(","):
            k, _, v = raw.partition("=")
            k = k.strip().lower()
            if not k:
                continue
            try:
                fv = float(v)
            except ValueError as e:
                raise ValueError(f"bad --dual-weights value: {raw!r}") from e
            if k in ("spatial", "spat"):
                spatial = fv if spatial is None else spatial
            elif k in ("spectral", "spec", "freq"):
                spectral = fv if spectral is None else spectral
            else:
                raise ValueError(f"unknown dual_weights key: {k!r}")
    return (spatial if spatial is not None else 1.0,
            spectral if spectral is not None else 1.0)


def _default_data_paths() -> Dict[str, str]:
    """Resolve data paths via project_paths (honours PROJECT_DATA_ROOT)."""
    from project_paths import Paths
    p = Paths()
    # metadata: prefer data root, fall back to repo copy
    meta = p.metadata if p.metadata.exists() else (DECODING_ROOT.parent / "encoding" / "data" / "metadata.json")
    return {
        "metadata": str(meta),
        "images": str(p.images_dir),
        "splits": str(DECODING_ROOT / "data" / "splits.json"),
    }


def _format_dry_run(
    plan: List[Tuple[ModelSpec, ResolvedCheckpoint]],
    args: argparse.Namespace,
    dual_w: Tuple[float, float],
) -> str:
    lines: List[str] = []
    lines.append("=== test (dry-run) ===")
    lines.append(f"  device       : {pick_device(args.device)}")
    lines.append(f"  weights_root : {args.weights_root}")
    lines.append(f"  metadata     : {args.metadata}")
    lines.append(f"  images       : {args.images}")
    lines.append(f"  splits       : {args.splits}  (split={args.split})")
    lines.append(f"  batch_size   : {args.batch_size}")
    lines.append(f"  num_workers  : {args.num_workers}")
    lines.append(f"  threshold    : {args.threshold}")
    lines.append(f"  dual weights : spatial={dual_w[0]} spectral={dual_w[1]}")
    lines.append("")
    for spec, ck in plan:
        cfg = spec.config or "(none committed)"
        status = "OK" if ck.path else "MISSING"
        path = str(ck.path) if ck.path else "n/a"
        lines.append(f"  - {spec.name:<22} cfg={cfg:<32} ckpt={status:<8} {path}")
        if not ck.path:
            for cand in ck.candidates_tried:
                lines.append(f"        tried: {cand}")
    return "\n".join(lines)


def _build_loader(args: argparse.Namespace, image_size: int):
    """Construct the evaluation DataLoader. Torch imports happen here so
    `--help` and `--dry-run` stay torch-free."""
    import json as _json
    import torch  # noqa: F401
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from src.dataloader import WatermarkDataset  # type: ignore

    paths = _default_data_paths()
    metadata = args.metadata or paths["metadata"]
    images = args.images or paths["images"]
    splits_path = args.splits or paths["splits"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full = WatermarkDataset(metadata_path=metadata, image_dir=images, transform=transform)
    with open(splits_path) as f:
        split_idx = _json.load(f)
    if args.split not in split_idx:
        raise ValueError(f"split {args.split!r} not present in {splits_path}")
    idxs = split_idx[args.split]
    if args.limit:
        idxs = idxs[: args.limit]
    subset = Subset(full, idxs)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    return loader, len(subset)


def _eval_one(
    spec: ModelSpec,
    ck: ResolvedCheckpoint,
    loader,
    device: str,
    threshold: float,
    dual_w: Tuple[float, float],
) -> Dict:
    import torch
    from src.models import get_model  # type: ignore
    from src.utils import compute_metrics  # type: ignore

    model = get_model(spec.name, num_outputs=8, pretrained=False)
    raw = torch.load(ck.path, map_location=device, weights_only=False)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(state)
    model.to(device).eval()

    ckpt_epoch = raw.get("epoch") if isinstance(raw, dict) else None
    ckpt_val = (raw.get("metrics") or {}).get("exact_match_rate") if isinstance(raw, dict) else None

    use_branch_weights = (
        spec.supports_branch_weights and (dual_w[0] != 1.0 or dual_w[1] != 1.0)
    )

    all_preds, all_targets = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            if use_branch_weights:
                logits = model.forward_with_branch_weights(
                    x, spatial_weight=dual_w[0], spectral_weight=dual_w[1]
                )
            else:
                logits = model(x)
            preds = (torch.sigmoid(logits) > threshold).float().cpu()
            all_preds.append(preds)
            all_targets.append(batch["bits"])
    elapsed = time.time() - t0
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)

    return {
        "model": spec.name,
        "checkpoint": str(ck.path),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_val_exact_match": ckpt_val,
        "image_size_used": loader.dataset.dataset.transform.transforms[0].size,  # type: ignore[attr-defined]
        "split_size": len(loader.dataset),
        "elapsed_sec": elapsed,
        "threshold": threshold,
        "branch_weights": {
            "spatial": dual_w[0],
            "spectral": dual_w[1],
            "applied": use_branch_weights,
        } if spec.supports_branch_weights else None,
        "metrics": {
            "per_bit_accuracy": [float(x) for x in metrics["per_bit_accuracy"]],
            "mean_bit_accuracy": float(metrics["mean_bit_accuracy"]),
            "exact_match_rate": float(metrics["exact_match_rate"]),
        },
    }


def _write_summary(output: Path, run_meta: Dict, per_model: List[Dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(
        {"run": run_meta, "models": per_model}, indent=2
    ))

    lines: List[str] = []
    lines.append("# Decoder evaluation summary")
    lines.append("")
    lines.append(f"- timestamp: `{run_meta['timestamp']}`")
    lines.append(f"- device: `{run_meta['device']}`")
    lines.append(f"- weights_root: `{run_meta['weights_root']}`")
    lines.append(f"- split: `{run_meta['split']}`  size={run_meta.get('split_size')}")
    if run_meta.get("dual_weights"):
        dw = run_meta["dual_weights"]
        lines.append(f"- dual weights: spatial={dw['spatial']} spectral={dw['spectral']}")
    lines.append("")
    lines.append("| model | mean_bit | exact | per-bit |")
    lines.append("| ----- | -------- | ----- | ------- |")
    for r in per_model:
        if "metrics" not in r:
            lines.append(f"| {r['model']} | _skipped_ | _{r.get('error','no ckpt')}_ |  |")
            continue
        m = r["metrics"]
        per = ",".join(f"{x:.3f}" for x in m["per_bit_accuracy"])
        lines.append(f"| {r['model']} | {m['mean_bit_accuracy']:.4f} | {m['exact_match_rate']:.4f} | {per} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    specs = parse_models_arg(args.models)
    overrides = parse_overrides(args.checkpoint)
    weights_root = Path(args.weights_root).expanduser()
    plan = [(s, resolve_checkpoint(s, weights_root, overrides)) for s in specs]

    dual_w = _parse_dual_weights(args)
    print(_format_dry_run(plan, args, dual_w))
    if args.dry_run:
        return 0

    missing = [s.name for s, ck in plan if ck.path is None]
    if missing:
        print(f"\nERROR: missing checkpoints for: {', '.join(missing)}")
        print("Pass --weights-root, --checkpoint NAME=PATH, or use --dry-run.")
        return 2

    device = pick_device(args.device)
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")

    per_model: List[Dict] = []
    split_size = 0
    for spec, ck in plan:
        image_size = args.image_size or spec.image_size
        loader, n = _build_loader(args, image_size)
        split_size = n
        print(f"\n--- evaluating {spec.name} on {n} images @ {image_size}px ---")
        try:
            result = _eval_one(spec, ck, loader, device, args.threshold, dual_w)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            per_model.append({"model": spec.name, "checkpoint": str(ck.path), "error": str(e)})
            continue
        m = result["metrics"]
        print(f"  mean_bit={m['mean_bit_accuracy']:.4f}  exact={m['exact_match_rate']:.4f}")
        per_model.append(result)
        # Per-model JSON for downstream tooling.
        (output / f"{spec.name}.json").write_text(json.dumps(result, indent=2))

    paths = _default_data_paths()
    run_meta = {
        "timestamp": timestamp,
        "device": device,
        "weights_root": str(weights_root),
        "metadata": args.metadata or paths["metadata"],
        "images": args.images or paths["images"],
        "splits": args.splits,
        "split": args.split,
        "split_size": split_size,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "dual_weights": {"spatial": dual_w[0], "spectral": dual_w[1]},
    }
    _write_summary(output, run_meta, per_model)
    print(f"\nwrote {output / 'summary.json'}")
    print(f"wrote {output / 'summary.md'}")
    return 0
