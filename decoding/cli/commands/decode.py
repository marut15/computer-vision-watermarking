"""`decoding.cli decode` — run a trained decoder on watermarked images."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import DECODING_ROOT, MODEL_BUNDLES_ROOT
from ..checkpoints import parse_overrides, resolve_checkpoint
from ..devices import pick_device
from ..registry import ModelSpec, all_model_names, resolve as resolve_model


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="Model name or alias. "
                        f"Known: {', '.join(all_model_names())}")
    p.add_argument("--input", required=True,
                   help="Image file or directory (recursive).")
    p.add_argument("--output", required=True,
                   help="Output directory for decoded results.")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint file. Overrides --weights-root.")
    p.add_argument("--weights-root", default=str(MODEL_BUNDLES_ROOT))
    p.add_argument("--s3-uri", default=None,
                   help="S3 URI to a checkpoint, used iff --download-missing.")
    p.add_argument("--download-missing", action="store_true",
                   help="If the checkpoint is missing locally, fetch from --s3-uri.")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=None,
                   help="Override the per-model default input size.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--format", default="json",
                   help="Comma-separated subset of: json,csv,md (default: json).")
    p.add_argument("--dual-spatial-weight", type=float, default=1.0)
    p.add_argument("--dual-spectral-weight", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")


def _collect_images(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMG_EXTS else []
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def _ensure_checkpoint(
    spec: ModelSpec, args: argparse.Namespace
) -> Tuple[Path, List[Path]]:
    overrides = parse_overrides([f"{spec.name}={args.checkpoint}"]) if args.checkpoint else {}
    weights_root = Path(args.weights_root).expanduser()
    ck = resolve_checkpoint(spec, weights_root, overrides)
    tried = list(ck.candidates_tried)
    if ck.path and ck.path.is_file():
        return ck.path, tried
    if args.s3_uri and args.download_missing:
        from ..s3 import download_s3_object
        # Drop the downloaded checkpoint into <weights-root>/<arch>/<arch>.pth.
        dest = weights_root / spec.name / spec.checkpoint_filename
        print(f"  downloading {args.s3_uri} -> {dest}")
        download_s3_object(args.s3_uri, dest)
        return dest, tried + [dest]
    raise FileNotFoundError(
        f"no checkpoint for {spec.name}; tried: " + ", ".join(str(p) for p in tried)
    )


def _load_image(path: Path, image_size: int):
    from PIL import Image  # type: ignore
    from torchvision import transforms  # type: ignore
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)


def _format_dry_run(spec: ModelSpec, ckpt: Optional[Path], imgs: List[Path],
                    args: argparse.Namespace) -> str:
    lines: List[str] = []
    lines.append("=== decode (dry-run) ===")
    lines.append(f"  model       : {spec.name}")
    lines.append(f"  checkpoint  : {ckpt or '(MISSING)'}")
    lines.append(f"  device      : {pick_device(args.device)}")
    lines.append(f"  image_size  : {args.image_size or spec.image_size}")
    lines.append(f"  input       : {args.input}")
    lines.append(f"  output      : {args.output}")
    lines.append(f"  n_images    : {len(imgs)}")
    if spec.supports_branch_weights:
        lines.append(f"  dual weights: spatial={args.dual_spatial_weight} "
                     f"spectral={args.dual_spectral_weight}")
    lines.append(f"  formats     : {args.format}")
    if imgs[:5]:
        for p in imgs[:5]:
            lines.append(f"    - {p}")
        if len(imgs) > 5:
            lines.append(f"    ... ({len(imgs) - 5} more)")
    return "\n".join(lines)


def _write_outputs(
    output_dir: Path, run_meta: Dict, records: List[Dict], formats: List[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if "json" in formats:
        (output_dir / "decoded.json").write_text(json.dumps(
            {"run": run_meta, "results": records}, indent=2
        ))
    if "csv" in formats:
        path = output_dir / "decoded.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "model", "checkpoint", "threshold",
                        *[f"bit{i}" for i in range(8)],
                        *[f"prob{i}" for i in range(8)],
                        "status"])
            for r in records:
                bits = r.get("predicted_bits") or [""] * 8
                probs = r.get("probabilities") or [""] * 8
                w.writerow([r["filename"], r["model"], r["checkpoint"],
                            r.get("threshold", ""), *bits, *probs,
                            r.get("status", "ok")])
    if "md" in formats:
        lines: List[str] = []
        lines.append(f"# Decode summary — {run_meta['model']}")
        lines.append("")
        lines.append(f"- checkpoint: `{run_meta['checkpoint']}`")
        lines.append(f"- device: `{run_meta['device']}`  threshold: {run_meta['threshold']}")
        lines.append(f"- n_images: {run_meta['n_images']}  "
                     f"failed: {run_meta['n_failed']}")
        lines.append("")
        lines.append("| filename | bits | status |")
        lines.append("| -------- | ---- | ------ |")
        for r in records:
            bits = "".join(str(b) for b in (r.get("predicted_bits") or []))
            lines.append(f"| `{r['filename']}` | `{bits or '-'}` | {r.get('status','ok')} |")
        (output_dir / "decoded.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    spec = resolve_model(args.model)
    input_root = Path(args.input).expanduser()
    if not input_root.exists():
        print(f"ERROR: --input does not exist: {input_root}")
        return 2
    images = _collect_images(input_root)

    formats = [t.strip().lower() for t in args.format.split(",") if t.strip()]
    bad = [t for t in formats if t not in {"json", "csv", "md"}]
    if bad:
        print(f"ERROR: unknown --format token(s): {bad}; expected json,csv,md")
        return 2

    # Resolve checkpoint location early so dry-run prints something useful.
    ckpt_candidates = []
    ckpt_path: Optional[Path] = None
    try:
        ckpt_path, ckpt_candidates = _ensure_checkpoint(spec, args)
    except FileNotFoundError as e:
        if not args.dry_run:
            print(f"ERROR: {e}")
            return 2

    print(_format_dry_run(spec, ckpt_path, images, args))
    if args.dry_run:
        return 0
    if not images:
        print("ERROR: no images found under --input.")
        return 2

    device = pick_device(args.device)
    image_size = args.image_size or spec.image_size

    import torch
    from src.models import get_model  # type: ignore

    model = get_model(spec.name, num_outputs=8, pretrained=False)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(state)
    model.to(device).eval()

    use_branch_weights = (
        spec.supports_branch_weights
        and (args.dual_spatial_weight != 1.0 or args.dual_spectral_weight != 1.0)
    )

    records: List[Dict] = []
    n_failed = 0
    timestamp = time.strftime("%Y%m%dT%H%M%S")

    batch: List[Tuple[Path, "torch.Tensor"]] = []  # (path, tensor)

    def _flush(batch_buf: List[Tuple[Path, "torch.Tensor"]]) -> None:
        if not batch_buf:
            return
        paths = [b[0] for b in batch_buf]
        x = torch.stack([b[1] for b in batch_buf]).to(device)
        with torch.no_grad():
            if use_branch_weights:
                logits = model.forward_with_branch_weights(
                    x, spatial_weight=args.dual_spatial_weight,
                    spectral_weight=args.dual_spectral_weight,
                )
            else:
                logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > args.threshold).int()
        logits_cpu = logits.detach().cpu().tolist()
        probs_cpu = probs.detach().cpu().tolist()
        preds_cpu = preds.detach().cpu().tolist()
        for p, pl, pp, pr in zip(paths, logits_cpu, probs_cpu, preds_cpu):
            records.append({
                "filename": str(p.relative_to(input_root)) if input_root.is_dir() else p.name,
                "abs_path": str(p),
                "model": spec.name,
                "checkpoint": str(ckpt_path),
                "threshold": args.threshold,
                "predicted_bits": pr,
                "probabilities": [float(v) for v in pp],
                "logits": [float(v) for v in pl],
                "status": "ok",
                "timestamp": timestamp,
                "branch_weights": (
                    {"spatial": args.dual_spatial_weight,
                     "spectral": args.dual_spectral_weight}
                    if use_branch_weights else None
                ),
            })

    for p in images:
        try:
            t = _load_image(p, image_size)
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            records.append({
                "filename": str(p.relative_to(input_root)) if input_root.is_dir() else p.name,
                "abs_path": str(p),
                "model": spec.name,
                "checkpoint": str(ckpt_path),
                "status": f"load_error: {e}",
                "timestamp": timestamp,
            })
            continue
        batch.append((p, t))
        if len(batch) >= args.batch_size:
            _flush(batch)
            batch = []
    _flush(batch)

    run_meta = {
        "model": spec.name,
        "checkpoint": str(ckpt_path),
        "device": device,
        "image_size": image_size,
        "threshold": args.threshold,
        "n_images": len(images),
        "n_failed": n_failed,
        "timestamp": timestamp,
        "input": str(input_root),
        "branch_weights": (
            {"spatial": args.dual_spatial_weight,
             "spectral": args.dual_spectral_weight}
            if use_branch_weights else None
        ),
    }

    output_dir = Path(args.output).expanduser()
    _write_outputs(output_dir, run_meta, records, formats)
    print(f"decoded {len(images) - n_failed}/{len(images)} images")
    print(f"wrote outputs under {output_dir}")
    if n_failed:
        print(f"warning: {n_failed} images failed to load (see status field).")
    return 0
