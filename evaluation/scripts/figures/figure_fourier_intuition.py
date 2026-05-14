"""Generate the 'Fourier intuition' figure for the deck.

Three panels stacked horizontally:
  (a) the watermarked image g  -  looks normal to the eye
  (b) Delta = g - f in pixel space  -  looks like structureless noise
  (c) log |F(Delta)|  -  energy concentrated in a few low-frequency bins

This is the matched-filter justification for DualBranch's spectral side:
the watermark is invisible per-pixel and obvious per-FFT-bin. Same
visualisation as Cox-Kilian-Leighton-Shamoon (1997, Fig. 1) for
spread-spectrum watermarking.

Usage:
  python evaluation/scripts/figures/figure_fourier_intuition.py
  python evaluation/scripts/figures/figure_fourier_intuition.py --prompt-id 3
  python evaluation/scripts/figures/figure_fourier_intuition.py --fft-crop 0   # full spectrum
  python evaluation/scripts/figures/figure_fourier_intuition.py \\
    --watermarked-image path/to/wm.png --baseline-image path/to/bl.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------- io helpers ---------------------------- #

def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def baseline_for_prompt(baseline_dir: Path, prompt_idx: int) -> Path | None:
    """Mirrors evaluation/scripts/figures/bit_difference_viewer.py path resolution."""
    candidates = [
        baseline_dir / f"prompt_{prompt_idx:02d}" / "baseline.png",
        baseline_dir / f"baseline_p{prompt_idx:02d}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def prompt_idx_from_entry(entry: dict[str, Any]) -> int | None:
    if "prompt_idx" in entry:
        return int(entry["prompt_idx"])
    f = str(entry.get("file", ""))
    if "_p" in f:
        digits = ""
        for ch in f.split("_p", 1)[1]:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return int(digits)
    return None


def find_pair(metadata_path: Path, images_dir: Path, baseline_dir: Path,
              prompt_idx: int):
    """Pick a watermarked image with maximum bits-set-to-1 at the given prompt
    (so the FFT panel shows as much structure as possible) and resolve its
    matching baseline."""
    metadata = json.loads(metadata_path.read_text())

    candidates: list[dict[str, Any]] = []
    for e in metadata:
        p = prompt_idx_from_entry(e)
        if p == prompt_idx:
            candidates.append(e)
    if not candidates:
        raise SystemExit(f"no metadata entries match prompt index {prompt_idx}")

    # Prefer the entry with the most active sliders (more energy in Delta).
    candidates.sort(key=lambda e: sum(e.get("bits") or [0]), reverse=True)
    chosen = candidates[0]
    wpath = images_dir / chosen["file"]
    bpath = baseline_for_prompt(baseline_dir, prompt_idx)
    if bpath is None:
        raise SystemExit(
            f"no baseline image found for prompt {prompt_idx} in {baseline_dir}"
        )
    return wpath, bpath, chosen.get("bits"), chosen.get("prompt", "")


# ---------------------------- figure ---------------------------- #

def make_figure(
    watermarked: np.ndarray,
    baseline: np.ndarray,
    bits,
    prompt: str,
    out_path: Path,
    fft_crop: int = 64,
) -> None:
    """Render the three-panel figure."""
    delta = watermarked - baseline               # H x W x 3 in [-1, 1]
    delta_gray = delta.mean(axis=-1)             # collapse RGB -> 2-D

    F = np.fft.fftshift(np.fft.fft2(delta_gray))
    fft_mag = np.log1p(np.abs(F))

    H, W = fft_mag.shape
    cy, cx = H // 2, W // 2

    if fft_crop > 0:
        r = min(fft_crop, H // 2, W // 2)
        fft_view = fft_mag[cy - r : cy + r, cx - r : cx + r]
        # Energy fraction inside the cropped low-frequency region.
        full_energy = float((np.abs(F) ** 2).sum())
        crop_energy = float(
            (np.abs(F[cy - r : cy + r, cx - r : cx + r]) ** 2).sum()
        )
        frac = crop_energy / max(full_energy, 1e-12)
        crop_label = f"central ${2 * r}^2$ bins, {frac * 100:.1f}% of $\\|\\hat{{\\Delta}}\\|^2$"
    else:
        fft_view = fft_mag
        crop_label = "full spectrum"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    # (a) watermarked image
    axes[0].imshow(np.clip(watermarked, 0, 1))
    axes[0].set_title("(a) watermarked image $g$\n(looks normal)",
                      fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # (b) Delta in pixel space
    vmax = max(0.02, float(np.percentile(np.abs(delta_gray), 99.5)))
    im_b = axes[1].imshow(delta_gray, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    linf = float(np.abs(delta_gray).max())
    axes[1].set_title(
        f"(b) $\\Delta = g - f$ in pixel space\n"
        f"$\\|\\Delta\\|_\\infty = {linf:.3f}$  (looks like noise)",
        fontsize=12, fontweight="bold",
    )
    axes[1].axis("off")
    cbar_b = fig.colorbar(im_b, ax=axes[1], fraction=0.04, pad=0.02)
    cbar_b.set_label("$\\Delta$", rotation=0, labelpad=10)

    # (c) log magnitude FFT
    im_c = axes[2].imshow(fft_view, cmap="magma")
    axes[2].set_title(
        f"(c) $\\log\\,|\\hat{{\\Delta}}|$ in frequency space\n"
        f"({crop_label})",
        fontsize=12, fontweight="bold",
    )
    axes[2].axis("off")
    fig.colorbar(im_c, ax=axes[2], fraction=0.04, pad=0.02)

    suptitle = "The watermark is a Fourier series  -  the FFT inverts the sum"
    if bits:
        suptitle += f"\n(bits = {bits})"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out_path}")


# ---------------------------- main ---------------------------- #

def main() -> None:
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parents[2]
    DECODING_ROOT = REPO / "decoding"

    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--metadata", default=None,
                   help="metadata.json path (default: watermark_encoding/data/metadata.json)")
    p.add_argument("--images", default=None,
                   help="watermarked images dir (default: watermark_encoding/data/images)")
    p.add_argument("--baseline", default=None,
                   help="baseline images dir (default: watermark_encoding/data/baseline)")
    p.add_argument("--watermarked-image", default=None,
                   help="explicit watermarked image path (overrides --metadata lookup)")
    p.add_argument("--baseline-image", default=None,
                   help="explicit baseline image path")
    p.add_argument("--prompt-id", type=int, default=0,
                   help="prompt index to use (0-based, default 0)")
    p.add_argument("--output", default=None,
                   help="output PNG path (default: <figures-root>/fourier_intuition.png)")
    p.add_argument("--fft-crop", type=int, default=64,
                   help="zoom radius for the FFT panel in bins around DC; 0 = full spectrum (default 64)")
    args = p.parse_args()

    metadata = Path(args.metadata) if args.metadata else REPO / "watermark_encoding" / "data" / "metadata.json"
    images_dir = Path(args.images) if args.images else REPO / "watermark_encoding" / "data" / "images"
    baseline_dir = Path(args.baseline) if args.baseline else REPO / "watermark_encoding" / "data" / "baseline"

    if args.output:
        output = Path(args.output)
    else:
        figures_root = Path("/workspace/new_models_figures") if Path("/workspace").is_dir() else REPO / "new_models_figures"
        output = figures_root / "fourier_intuition.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.watermarked_image and args.baseline_image:
        wpath = Path(args.watermarked_image)
        bpath = Path(args.baseline_image)
        bits = None
        prompt = ""
    else:
        wpath, bpath, bits, prompt = find_pair(metadata, images_dir, baseline_dir, args.prompt_id)

    print(f"[figure] watermarked: {wpath}")
    print(f"[figure] baseline:    {bpath}")
    if bits is not None:
        print(f"[figure] bits:        {bits}")
    if prompt:
        print(f"[figure] prompt:      {prompt}")

    watermarked = load_image(wpath)
    baseline = load_image(bpath)

    # Resize baseline to the watermarked resolution if they differ.
    if watermarked.shape != baseline.shape:
        target_h, target_w = watermarked.shape[:2]
        baseline_pil = Image.fromarray((baseline * 255.0).astype(np.uint8))
        baseline_pil = baseline_pil.resize((target_w, target_h), Image.BICUBIC)
        baseline = np.asarray(baseline_pil, dtype=np.float32) / 255.0

    make_figure(watermarked, baseline, bits, prompt, output, fft_crop=args.fft_crop)


if __name__ == "__main__":
    main()
