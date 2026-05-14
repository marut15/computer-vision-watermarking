"""Synthetic-control experiment: what does a spatially-localised broadband
Delta look like in the spectrum the spectral branch actually consumes?

The puzzle figure shows Delta = (g - f) at 1024^2 with two properties:
  - spatially localised in pixel space (high-texture regions only)
  - broadband in frequency space (1.5% energy in central 128^2 of 1024^2)

But DualBranch's spectral branch does NOT see |F(Delta)| at 1024. It sees
``log1p(|F(g_pool)|)'' standardised per-image, where g_pool is the image
adaptive-avg-pooled to 256x256. That is a very different observation.

This script constructs a controlled Delta with the same first-order
properties and shows what the model's spectral input looks like:

  1. Build a synthetic ``baseline'' f as a 1/f^alpha colour noise field
     (good first-order match to natural-image statistics, and we don't need
     external data). Optionally read a real PNG via --image PATH.
  2. Compute a smooth spatial mask M(x, y) that concentrates Delta in
     high-variance regions (mimicking ``texture-rich pixels only'').
  3. Sample a broadband white-noise field W and form Delta = epsilon * M * W.
     Two ``bit'' settings: bit=0 (epsilon = 0) and bit=1 (epsilon > 0).
  4. Form g = f + Delta. Compare:
       - log1p|F(f)|  (standardised, 256x256, after avg-pool)
       - log1p|F(g)|  (standardised, 256x256)
       - their difference: this is what the spectral branch CAN read.

The output figure mirrors the puzzle figure but adds two crucial new panels
on the model's actual input space (256x256 standardised log spectrum):

  (a) f, (b) Delta in pixel space, (c) log|F(Delta)| at 1024,
  (d) standardised log|F(f)|_256, (e) standardised log|F(g)|_256,
  (f) (e)-(d): what the model can possibly see.

If (f) is broadband and image-content-modulated, that is direct evidence
that the spectral CNN must be reading an envelope / image-content-sensitive
feature, not a narrow Fourier peak.

Usage:
  python decoding/scripts/synthetic_broadband_control.py
  python decoding/scripts/synthetic_broadband_control.py --image path/to/baseline.png
  python decoding/scripts/synthetic_broadband_control.py --epsilon 0.05

Outputs:
  - decoding/results/figures/synthetic_broadband_control.png
  - decoding/results/synthetic_broadband_control.json (numerical summary)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent


def make_color_noise(size: int, alpha: float = 1.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size, 3), dtype=np.float32)
    yy, xx = np.indices((size, size))
    cy = cx = size // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    decay = 1.0 / (rr + 1) ** (alpha / 2.0)
    decay = np.fft.ifftshift(decay)
    for c in range(3):
        white = rng.standard_normal((size, size)).astype(np.float32)
        F_w = np.fft.fft2(white) * decay
        chan = np.real(np.fft.ifft2(F_w))
        chan = chan - chan.min()
        chan = chan / max(chan.max(), 1e-8)
        out[..., c] = chan
    # add a smooth structure to mimic an image: a low-rank colour gradient
    grad = np.linspace(0, 1, size, dtype=np.float32)
    bg = np.stack(
        [
            0.4 + 0.2 * grad[None, :].repeat(size, 0),
            0.3 + 0.3 * (grad[:, None] * grad[None, :]),
            0.2 + 0.5 * grad[:, None].repeat(size, 1),
        ],
        axis=-1,
    )
    return np.clip(0.55 * out + 0.45 * bg, 0, 1)


def texture_mask(image: np.ndarray, kernel_size: int = 24) -> np.ndarray:
    """Box-averaged local std of grayscale via torch conv: high in textured regions."""
    g = image.mean(axis=-1)
    t = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    k = max(int(kernel_size), 3)
    pad = k // 2
    if k % 2 == 0:
        k += 1
        pad = k // 2
    kernel = torch.full((1, 1, k, k), 1.0 / (k * k))
    mean = F.conv2d(F.pad(t, (pad, pad, pad, pad), mode="reflect"), kernel)
    mean2 = F.conv2d(F.pad(t ** 2, (pad, pad, pad, pad), mode="reflect"), kernel)
    var = (mean2 - mean ** 2).clamp_min(0.0)
    std = var.sqrt().squeeze().numpy()
    if std.max() > 0:
        std /= std.max()
    return std.astype(np.float32)


def standardise_log_spec(image_hwc: np.ndarray, fft_size: int) -> np.ndarray:
    """Return per-channel-standardised log1p|F(pool(image))| matching the
    model's _spectrum (averaged across channels for visualisation)."""
    t = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float()
    t = F.adaptive_avg_pool2d(t, fft_size)
    spec = torch.log1p(torch.fft.fftshift(torch.fft.fft2(t).abs(), dim=(-1, -2)))
    mu = spec.mean(dim=(-1, -2), keepdim=True)
    sigma = spec.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    spec = (spec - mu) / sigma
    return spec.squeeze(0).mean(0).numpy()


def central_energy_fraction(power_2d: np.ndarray, r: int) -> float:
    h, w = power_2d.shape
    cy, cx = h // 2, w // 2
    return float(power_2d[cy - r : cy + r, cx - r : cx + r].sum() / max(power_2d.sum(), 1e-12))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--image", default=None, help="Path to a baseline PNG. Default: synthetic 1/f^alpha noise.")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--epsilon", type=float, default=0.06,
                   help="Magnitude of the synthetic Delta (mean over the texture mask).")
    p.add_argument("--alpha", type=float, default=1.5,
                   help="1/f^alpha exponent for synthetic baseline.")
    p.add_argument("--fft-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-fig", default=str(DECODING_ROOT / "results" / "figures" / "synthetic_broadband_control.png"))
    p.add_argument("--out-json", default=str(DECODING_ROOT / "results" / "synthetic_broadband_control.json"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.image:
        f = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0
        if f.shape[0] != args.size:
            f = np.asarray(
                Image.fromarray((f * 255).astype(np.uint8)).resize(
                    (args.size, args.size), Image.BICUBIC
                )
            ) / 255.0
    else:
        f = make_color_noise(args.size, alpha=args.alpha, seed=args.seed)

    rng = np.random.default_rng(args.seed + 1)
    mask = texture_mask(f, kernel_size=24)
    white = rng.standard_normal((args.size, args.size)).astype(np.float32)
    delta_unit = mask * white
    # Normalise so the *masked* region has a controlled std.
    target_std = args.epsilon
    cur_std = delta_unit[mask > 0.05].std() or 1.0
    delta = (target_std / cur_std) * delta_unit
    delta_rgb = np.stack([delta] * 3, axis=-1)
    g = np.clip(f + delta_rgb, 0, 1)

    delta_gray = (g - f).mean(axis=-1)

    F_full = np.fft.fftshift(np.fft.fft2(delta_gray))
    fft_full_logmag = np.log1p(np.abs(F_full))
    pow_d_full = np.abs(F_full) ** 2

    spec_f = standardise_log_spec(f, args.fft_size)
    spec_g = standardise_log_spec(g, args.fft_size)
    spec_diff = spec_g - spec_f
    pow_g_pool = np.abs(np.fft.fftshift(np.fft.fft2(F.adaptive_avg_pool2d(
        torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).float(), args.fft_size
    ).mean(1).squeeze().numpy())))
    # ^ for measurement only; not used for figure

    summary = {
        "delta_linf": float(np.abs(delta_gray).max()),
        "delta_l2_rms": float(np.sqrt((delta_gray ** 2).mean())),
        "energy_frac_central_128_full": central_energy_fraction(pow_d_full, 128),
        "energy_frac_central_64_full": central_energy_fraction(pow_d_full, 64),
        "uniform_baseline_central_128": (2 * 128) ** 2 / (args.size ** 2),
        "spec_diff_linf": float(np.abs(spec_diff).max()),
        "spec_diff_rms": float(np.sqrt((spec_diff ** 2).mean())),
        "spec_diff_energy_frac_central_64": central_energy_fraction(spec_diff ** 2, 64),
        "spec_diff_energy_frac_central_32": central_energy_fraction(spec_diff ** 2, 32),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Six-panel figure.
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(f); axes[0, 0].set_title("(a) baseline $f$"); axes[0, 0].axis("off")
    vmax = max(0.02, float(np.percentile(np.abs(delta_gray), 99.5)))
    axes[0, 1].imshow(delta_gray, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title(
        f"(b) $\\Delta$ (texture-masked, $\\|\\Delta\\|_\\infty={summary['delta_linf']:.3f}$)"
    )
    axes[0, 1].axis("off")
    cy = args.size // 2
    crop = 128
    axes[0, 2].imshow(fft_full_logmag[cy - crop : cy + crop, cy - crop : cy + crop], cmap="magma")
    axes[0, 2].set_title(
        f"(c) $\\log|F(\\Delta)|_{{1024}}$  ({summary['energy_frac_central_128_full']*100:.1f}% in central $128^2$)"
    )
    axes[0, 2].axis("off")

    axes[1, 0].imshow(spec_f, cmap="magma"); axes[1, 0].set_title("(d) std. $\\log|F(f)|_{256}$"); axes[1, 0].axis("off")
    axes[1, 1].imshow(spec_g, cmap="magma"); axes[1, 1].set_title("(e) std. $\\log|F(g)|_{256}$"); axes[1, 1].axis("off")
    v = max(0.005, float(np.percentile(np.abs(spec_diff), 99.5)))
    im = axes[1, 2].imshow(spec_diff, cmap="seismic", vmin=-v, vmax=v)
    axes[1, 2].set_title(
        f"(f) (e) - (d): what the spectral CNN sees\n"
        f"linf={summary['spec_diff_linf']:.3f}  rms={summary['spec_diff_rms']:.4f}  "
        f"central-$32^2$ energy = {summary['spec_diff_energy_frac_central_32']*100:.1f}%"
    )
    axes[1, 2].axis("off")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.04, pad=0.02)
    fig.suptitle(
        "Synthetic broadband, texture-localised $\\Delta$:\n"
        "what the model's spectral branch ACTUALLY sees (per-image-standardised, $256^2$)",
        fontweight="bold",
    )
    fig.tight_layout()
    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[synthetic] figure -> {out_fig}")


if __name__ == "__main__":
    main()
