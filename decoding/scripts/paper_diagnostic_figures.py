"""Generate paper diagnostic figures for the dual_branch_r50 decoder.

Figures produced
----------------
paper_gradcam_templates_bits.png / .pdf
    Grad-CAM grid — rows = prompt indices, cols = 8 bits.
    Each cell shows the Grad-CAM overlay on the single-bit-active image
    (id_int = 2**(7-bit_idx)) for that prompt, with gradient taken on the
    raw logit of that bit.

paper_fft_baseline_vs_allbits.png / .pdf
    FFT comparison — rows = prompt indices,
    cols = [Baseline | All bits=1 | Baseline FFT | All-bits FFT | FFT diff].

Usage
-----
    python decoding/scripts/paper_diagnostic_figures.py \\
        --model dual_branch_r50 \\
        --output-dir decoding/results/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ── path setup ────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent   # decoding/scripts/
DECODING_ROOT = HERE.parent                       # decoding/
REPO_ROOT    = DECODING_ROOT.parent               # repo root

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DECODING_ROOT))

from project_paths import Paths
from src.models import get_model


# ── metadata helpers ──────────────────────────────────────────────────────────

def _load_metadata(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _index_by_id_prompt(metadata: list[dict]) -> dict[tuple[int, int], dict]:
    """Map (id_int, prompt_idx) -> entry. Prompt idx parsed from filename suffix _pXX."""
    index: dict[tuple[int, int], dict] = {}
    for entry in metadata:
        try:
            p_idx = int(Path(entry["file"]).stem.split("_p")[-1])
        except (ValueError, IndexError):
            continue
        index[(entry["id_int"], p_idx)] = entry
    return index


def _baseline_path(baseline_dir: Path, prompt_idx: int) -> Path:
    return baseline_dir / f"baseline_p{prompt_idx:02d}.png"


def _open_rgb(path: Path) -> np.ndarray:
    """Load image as HWC float32 in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _log_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """Log-magnitude 2D FFT of channel-mean grayscale, normalized to [0, 1]."""
    grey = image.mean(axis=-1)
    fft  = np.fft.fftshift(np.fft.fft2(grey))
    mag  = np.log1p(np.abs(fft))
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


# ── model utilities ───────────────────────────────────────────────────────────

def resolve_checkpoint(model_name: str, explicit: Optional[str], paths: Paths) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--checkpoint not found: {p}")
        return p
    candidates = [
        paths.model_bundles / model_name / f"{model_name}.pth",
        paths.model_bundles / model_name / "model.pth",
        paths.decoder_checkpoints / f"{model_name}.pth",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No checkpoint found for '{model_name}'. Tried:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def load_model(model_name: str, checkpoint: Path, device: torch.device) -> nn.Module:
    model = get_model(model_name, num_outputs=8, pretrained=False)
    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def _spatial_target_layer(model: nn.Module) -> nn.Module:
    """Last Conv2d-containing block in model.spatial before AdaptiveAvgPool2d/Flatten."""
    last = None
    for child in model.spatial.children():
        if isinstance(child, (nn.AdaptiveAvgPool2d, nn.Flatten)):
            break
        if any(isinstance(m, nn.Conv2d) for m in child.modules()):
            last = child
    if last is None:
        raise RuntimeError(
            "Could not find a Conv2d-containing layer in model.spatial before pooling"
        )
    return last


class _BitLogitTarget:
    """pytorch_grad_cam target: gradient w.r.t. raw logit for one bit index.

    pytorch_grad_cam calls targets per-sample with a 1D tensor, so we handle
    both shapes: (num_bits,) for single-sample calls and (B, num_bits) batched.
    """
    def __init__(self, bit_idx: int):
        self.bit_idx = bit_idx

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.dim() == 1:
            return model_output[self.bit_idx]
        return model_output[:, self.bit_idx]


def _missing_cell(ax: plt.Axes, label: str = "missing") -> None:
    ax.set_facecolor("#eeeeee")
    ax.text(0.5, 0.5, label, ha="center", va="center",
            fontsize=7, color="#777777", transform=ax.transAxes)
    ax.axis("off")


# ── Figure 1: Grad-CAM grid ───────────────────────────────────────────────────

def figure_gradcam(
    model: nn.Module,
    index: dict[tuple[int, int], dict],
    images_dir: Path,
    prompt_indices: list[int],
    image_size: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("[gradcam] pytorch_grad_cam not found — pip install grad-cam  (skipping)")
        return

    n_rows = len(prompt_indices)
    n_cols = 8

    to_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    to_display = transforms.Resize(
        (image_size, image_size),
        interpolation=transforms.InterpolationMode.LANCZOS,
    )

    target_layer = _spatial_target_layer(model)
    print(f"  [gradcam] target layer: {type(target_layer).__name__}")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.3),
        squeeze=False,
    )

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for row, p_idx in enumerate(prompt_indices):
            for bit_idx in range(n_cols):
                ax = axes[row][bit_idx]

                if row == 0:
                    ax.set_title(f"S{bit_idx + 1}", fontsize=9, fontweight="bold", pad=3)
                if bit_idx == 0:
                    ax.set_ylabel(f"P{p_idx}", fontsize=8, rotation=0,
                                  labelpad=22, va="center")

                entry = index.get((2 ** (7 - bit_idx), p_idx))
                if entry is None:
                    _missing_cell(ax)
                    continue

                img_path = images_dir / entry["file"]
                if not img_path.exists():
                    _missing_cell(ax)
                    continue

                try:
                    pil = Image.open(img_path).convert("RGB")
                    rgb_arr    = np.array(to_display(pil), dtype=np.float32) / 255.0
                    img_tensor = to_tensor(pil).unsqueeze(0).to(device)

                    grayscale_cam = cam(
                        input_tensor=img_tensor,
                        targets=[_BitLogitTarget(bit_idx)],
                    )
                    overlay = show_cam_on_image(rgb_arr, grayscale_cam[0], use_rgb=True)
                    ax.imshow(overlay)
                except Exception as exc:
                    _missing_cell(ax, label=f"error\n{type(exc).__name__}")
                    print(f"  [gradcam] P{p_idx} bit{bit_idx}: {exc}")

                ax.axis("off")

    fig.suptitle(
        "Grad-CAM on single-bit-active images  (rows = prompts, cols = sliders)",
        fontsize=10, y=1.002,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "paper_gradcam_templates_bits")


# ── Figure 2: FFT grid ────────────────────────────────────────────────────────

def figure_fft(
    index: dict[tuple[int, int], dict],
    images_dir: Path,
    baseline_dir: Path,
    prompt_indices: list[int],
    output_dir: Path,
) -> None:
    n_rows = len(prompt_indices)
    n_cols = 5
    col_labels = ["Baseline", "All bits = 1", "Baseline FFT", "All-bits FFT", "FFT difference"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.6),
        squeeze=False,
    )

    for row, p_idx in enumerate(prompt_indices):
        row_axes = axes[row]

        if row == 0:
            for col, label in enumerate(col_labels):
                row_axes[col].set_title(label, fontsize=9, fontweight="bold", pad=3)
        row_axes[0].set_ylabel(f"P{p_idx}", fontsize=8, rotation=0,
                               labelpad=22, va="center")

        bl_path       = _baseline_path(baseline_dir, p_idx)
        allbits_entry = index.get((255, p_idx))
        allbits_path  = (images_dir / allbits_entry["file"]) if allbits_entry else None

        baseline_ok = bl_path.exists()
        allbits_ok  = allbits_path is not None and allbits_path.exists()

        if not baseline_ok or not allbits_ok:
            for ax in row_axes:
                _missing_cell(ax)
            if not baseline_ok:
                print(f"  [fft] P{p_idx}: baseline not found at {bl_path}")
            if not allbits_ok:
                print(f"  [fft] P{p_idx}: all-bits image not found")
            continue

        try:
            bl_img = _open_rgb(bl_path)
            ab_img = _open_rgb(allbits_path)

            if ab_img.shape != bl_img.shape:
                h, w   = bl_img.shape[:2]
                ab_img = np.asarray(
                    Image.fromarray((ab_img * 255).astype(np.uint8))
                    .resize((w, h), Image.LANCZOS),
                    dtype=np.float32,
                ) / 255.0

            fft_bl  = _log_fft_magnitude(bl_img)
            fft_ab  = _log_fft_magnitude(ab_img)
            diff    = fft_ab - fft_bl
            abs_max = float(np.abs(diff).max()) or 1e-6

            row_axes[0].imshow(bl_img);                                         row_axes[0].axis("off")
            row_axes[1].imshow(ab_img);                                         row_axes[1].axis("off")
            row_axes[2].imshow(fft_bl, cmap="magma");                           row_axes[2].axis("off")
            row_axes[3].imshow(fft_ab, cmap="magma");                           row_axes[3].axis("off")
            row_axes[4].imshow(diff, cmap="seismic",
                               vmin=-abs_max, vmax=abs_max);                    row_axes[4].axis("off")

        except Exception as exc:
            for ax in row_axes:
                _missing_cell(ax, label=f"error\n{type(exc).__name__}")
            print(f"  [fft] P{p_idx}: {exc}")

    fig.suptitle(
        "FFT: baseline vs all-bits-on watermarked  (rows = prompts)",
        fontsize=10, y=1.002,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "paper_fft_baseline_vs_allbits")


# ── save helper ───────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    _p = Paths()
    ap = argparse.ArgumentParser(
        description="Generate paper diagnostic figures for a watermark decoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="dual_branch_r50",
                    help="Model architecture name (used for checkpoint lookup)")
    ap.add_argument("--checkpoint", default=None,
                    help="Explicit checkpoint path; auto-resolved if omitted")
    ap.add_argument("--metadata",     default=str(_p.metadata))
    ap.add_argument("--images",       default=str(_p.images_dir))
    ap.add_argument("--baseline-dir", default=str(_p.baseline_dir))
    ap.add_argument("--output-dir",   default=str(DECODING_ROOT / "results" / "figures"))
    ap.add_argument("--prompt-indices", default=None,
                    help="Comma-separated prompt indices, e.g. 0,1,2; default: all")
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="Limit number of prompts (applied after --prompt-indices)")
    ap.add_argument("--image-size", type=int, default=1024,
                    help="Resize images to this size before model/FFT")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--skip-gradcam", action="store_true")
    ap.add_argument("--skip-fft",     action="store_true")
    return ap.parse_args()


def main() -> None:
    args       = parse_args()
    paths      = Paths()

    metadata_path = Path(args.metadata)
    images_dir    = Path(args.images)
    baseline_dir  = Path(args.baseline_dir)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_checkpoint(args.model, args.checkpoint, paths)

    metadata       = _load_metadata(metadata_path)
    index          = _index_by_id_prompt(metadata)
    all_prompts    = sorted({p for (_, p) in index})
    prompt_indices = (
        [int(x) for x in args.prompt_indices.split(",")]
        if args.prompt_indices
        else all_prompts
    )
    if args.max_prompts:
        prompt_indices = prompt_indices[: args.max_prompts]

    device = resolve_device(args.device)

    print("=== paper_diagnostic_figures ===")
    print(f"  metadata     : {metadata_path}")
    print(f"  images       : {images_dir}")
    print(f"  baseline     : {baseline_dir}")
    print(f"  checkpoint   : {checkpoint}")
    print(f"  output       : {output_dir}")
    print(f"  prompts ({len(prompt_indices):3d}): {prompt_indices}")
    print(f"  model        : {args.model}")
    print(f"  device       : {device}")
    print(f"  image_size   : {args.image_size}")

    if not args.skip_gradcam:
        print("\n[1/2] loading model ...")
        model = load_model(args.model, checkpoint, device)
        print("[1/2] Grad-CAM grid ...")
        figure_gradcam(model, index, images_dir, prompt_indices,
                       args.image_size, device, output_dir)
    else:
        print("\n[1/2] Grad-CAM (skipped)")

    if not args.skip_fft:
        print("\n[2/2] FFT grid ...")
        figure_fft(index, images_dir, baseline_dir, prompt_indices, output_dir)
    else:
        print("\n[2/2] FFT (skipped)")

    print("\nDone.")


if __name__ == "__main__":
    main()
