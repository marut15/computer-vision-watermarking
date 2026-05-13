"""Synthetic-control experiment 2: a multiplicative spectral-envelope gain
generates a *spatially localised, broadband* Delta in pixel space.

Working hypothesis: the spectral CNN reads a multiplicative gain on |F(g)|
(applied during generation by the LoRA). This script tests whether a
multiplicative gain in frequency space *generates* the empirical
Delta-properties from the puzzle:

  - Delta = g - f looks spatially localised (energy in textured pixels)
  - Delta has a broadband-looking 1024^2 FFT
  - log|F(g)| - log|F(f)| has a clear radial structure that the spectral
    CNN can read with a tiny linear classifier on the radial profile

Construction:
  1. f = synthetic 1/f^alpha colour image (or read from --image).
  2. F = FFT(f) per channel.
  3. Apply a *radially-banded* multiplicative gain envelope:
     ``G(k) = 1 + bit * gain * smoothbump(k; k0, sigma)``
     -- larger gain in a frequency band centred at k0.
  4. g = clip(IFFT(G(k) * F), 0, 1).
  5. Plot the same six panels as synthetic_broadband_control.

If the empirical Delta from a multiplicative spectral gain looks
qualitatively like the puzzle figure (spatially localised + 1024^2 FFT
near-uniform), and (e)-(d) shows a clean radial bump, that supports the
hypothesis ``the spectral CNN reads a multiplicative envelope on |F(g)|''.

Outputs:
  - decoding/results/figures/synthetic_envelope_control.png
  - decoding/results/synthetic_envelope_control.json

Usage:
  python decoding/scripts/synthetic_envelope_control.py
  python decoding/scripts/synthetic_envelope_control.py --gain 0.15 --k0 60
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

# Reuse helpers from synthetic_broadband_control by importing them.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "synthetic_broadband_control",
    str(HERE / "synthetic_broadband_control.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore
make_color_noise = _mod.make_color_noise
standardise_log_spec = _mod.standardise_log_spec
central_energy_fraction = _mod.central_energy_fraction


def smoothbump(rr: np.ndarray, k0: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((rr - k0) / sigma) ** 2)


def apply_radial_gain(image: np.ndarray, k0: float, sigma: float, gain: float) -> np.ndarray:
    H, W, C = image.shape
    yy, xx = np.indices((H, W))
    cy, cx = H // 2, W // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    g_envelope = 1.0 + gain * smoothbump(rr, k0, sigma)
    g_envelope = np.fft.ifftshift(g_envelope)
    out = np.zeros_like(image)
    for c in range(C):
        Fc = np.fft.fft2(image[..., c])
        Gc = Fc * g_envelope
        out[..., c] = np.real(np.fft.ifft2(Gc))
    return np.clip(out, 0, 1).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--image", default=None)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--k0", type=float, default=60.0,
                   help="Centre radius (in 1024^2 FFT bins) of the gain bump.")
    p.add_argument("--sigma", type=float, default=12.0)
    p.add_argument("--gain", type=float, default=0.20,
                   help="Peak multiplicative gain (0.20 = +20%% at k0).")
    p.add_argument("--fft-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-fig", default=str(DECODING_ROOT / "results" / "figures" / "synthetic_envelope_control.png"))
    p.add_argument("--out-json", default=str(DECODING_ROOT / "results" / "synthetic_envelope_control.json"))
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

    g = apply_radial_gain(f, k0=args.k0, sigma=args.sigma, gain=args.gain)
    delta_gray = (g - f).mean(axis=-1)
    F_full = np.fft.fftshift(np.fft.fft2(delta_gray))
    fft_full_logmag = np.log1p(np.abs(F_full))
    pow_d_full = np.abs(F_full) ** 2

    spec_f = standardise_log_spec(f, args.fft_size)
    spec_g = standardise_log_spec(g, args.fft_size)
    spec_diff = spec_g - spec_f

    # Spatial localisation of |Delta|: top-decile fraction.
    abs_d = np.abs(delta_gray)
    e = abs_d ** 2
    sorted_e = np.sort(e.ravel())[::-1]
    top10 = float(sorted_e[: e.size // 10].sum() / max(e.sum(), 1e-12))

    summary = {
        "k0_1024": args.k0,
        "k0_256_eq": args.k0 / 4,  # because we pool by 4x
        "gain": args.gain,
        "delta_linf": float(abs_d.max()),
        "delta_l2_rms": float(np.sqrt(e.mean())),
        "top_decile_energy_frac_pixel": top10,
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
    vmax = max(0.005, float(np.percentile(np.abs(delta_gray), 99.5)))
    axes[0, 1].imshow(delta_gray, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title(
        f"(b) $\\Delta = g - f$  ($\\|\\Delta\\|_\\infty = {summary['delta_linf']:.3f}$,\n"
        f"top-decile energy = {top10*100:.0f}%)"
    )
    axes[0, 1].axis("off")
    cy = args.size // 2
    crop = 128
    axes[0, 2].imshow(fft_full_logmag[cy - crop : cy + crop, cy - crop : cy + crop], cmap="magma")
    axes[0, 2].set_title(
        f"(c) $\\log|F(\\Delta)|_{{1024}}$\n"
        f"({summary['energy_frac_central_128_full']*100:.1f}% in central $128^2$)"
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
        "Synthetic CONSTRUCTION: a multiplicative radial gain on $|F(f)|$ produces\n"
        "(b) a spatially-localised broadband $\\Delta$  AND  (f) a clean radial bump in std.$\\log|F(g)|$.",
        fontweight="bold",
    )
    fig.tight_layout()
    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[synthetic_envelope] figure -> {out_fig}")


if __name__ == "__main__":
    main()
