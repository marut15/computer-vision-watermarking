"""Characterise Delta = (watermarked - baseline) across the full dataset.

The puzzle: for one prompt, Delta is spatially localised to texture regions
yet has a broadband 1024^2 FFT (1.5% energy in the central 128^2 bins, vs
1.56% uniform). This script asks the same question across all (prompt, bits)
pairs and aggregates the answer.

For each prompt and each bit-set b in {0..255}:

  1. Load watermarked image g and baseline f at the same prompt.
  2. Compute Delta = g - f.
  3. Pixel-space stats:
       - L_inf, L2, fraction of pixels with |Delta| > 1/255 (visible-quantum)
       - spatial entropy of |Delta| (heuristic for 'spatially localised')
       - top-decile spatial concentration: fraction of total energy in the
         pixels with the top 10% |Delta| values
  4. Frequency-space stats on the SAME spectrum the model sees:
       - Adaptive-avg-pool Delta (and g, f) to 256x256 to match the model's
         spec_downsample, take 2D FFT, magnitude.
       - Radial profile P(k) = mean |F|^2 over annuli of width 1.
       - Energy fraction inside central 32^2 / 64^2 / 128^2 bins.
       - Anisotropy: ratio (horizontal-axis power) / (vertical-axis power).
  5. Per-bit aggregation: for each of the 8 slider bits, mean radial profile
     conditioned on bit_i = 1 vs bit_i = 0. Distance between the two profiles
     (KS, L2, log-ratio) is the candidate signal the spectral CNN reads.

Outputs:
  - decoding/results/delta_dataset.json  (all stats, indexed by entry)
  - decoding/results/figures/delta_dataset/
      mean_radial_profile.png
      per_bit_radial_profile.png
      energy_concentration_hist.png
      spatial_concentration_hist.png

Usage:
  python decoding/scripts/delta_dataset_analysis.py
  python decoding/scripts/delta_dataset_analysis.py --prompt-ids 0,1,2
  python decoding/scripts/delta_dataset_analysis.py --fft-size 256

This is pure analysis; it does not load the trained decoder.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent
REPO = DECODING_ROOT.parent


def _open_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _baseline_path(baseline_dir: Path, prompt_idx: int) -> Path | None:
    for c in (
        baseline_dir / f"prompt_{prompt_idx:02d}" / "baseline.png",
        baseline_dir / f"baseline_p{prompt_idx:02d}.png",
    ):
        if c.exists():
            return c
    return None


def _prompt_idx_from_entry(entry: dict) -> int | None:
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


def _avg_pool_to(arr_hwc: np.ndarray, size: int) -> np.ndarray:
    """Adaptive-avg-pool an (H, W, C) image to (size, size, C). Matches the
    model's ``nn.AdaptiveAvgPool2d(size)``."""
    t = torch.from_numpy(arr_hwc).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    t = F.adaptive_avg_pool2d(t, size)
    return t.squeeze(0).permute(1, 2, 0).numpy()


def _radial_profile(power_2d: np.ndarray) -> np.ndarray:
    """Mean power in annular bins of width 1, indexed by integer radius.

    ``power_2d`` is fftshifted (DC at centre)."""
    h, w = power_2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    nbins = min(cy, cx) + 1
    out = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)
    flat_r = r.ravel()
    flat_p = power_2d.ravel()
    mask = flat_r < nbins
    np.add.at(out, flat_r[mask], flat_p[mask])
    np.add.at(counts, flat_r[mask], 1)
    counts = np.maximum(counts, 1)
    return out / counts


def _spatial_top_decile_fraction(abs_delta_gray: np.ndarray) -> float:
    """Fraction of |Delta|^2 energy contained in the top 10% of pixels."""
    flat = abs_delta_gray.ravel()
    e = flat ** 2
    total = e.sum()
    if total <= 0:
        return 0.0
    sorted_e = np.sort(e)[::-1]
    cutoff = int(0.1 * len(sorted_e))
    return float(sorted_e[:cutoff].sum() / total)


def _spatial_entropy(abs_delta_gray: np.ndarray) -> float:
    """Shannon entropy of the per-pixel energy distribution, normalised by
    log(N). 1.0 = uniform (broadband), 0.0 = a single pixel."""
    e = abs_delta_gray.ravel() ** 2
    total = e.sum()
    if total <= 0:
        return 0.0
    p = e / total
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(H / np.log(len(e)))


def _central_energy_fraction(power_2d: np.ndarray, r: int) -> float:
    h, w = power_2d.shape
    cy, cx = h // 2, w // 2
    r = min(r, cy, cx)
    inner = power_2d[cy - r : cy + r, cx - r : cx + r].sum()
    total = power_2d.sum()
    if total <= 0:
        return 0.0
    return float(inner / total)


def _anisotropy(power_2d: np.ndarray, halfwidth: int = 2) -> float:
    """Ratio of mean power on the horizontal axis to vertical axis (centred
    +- halfwidth rows / columns around DC). > 1 = horizontal-dominant."""
    h, w = power_2d.shape
    cy, cx = h // 2, w // 2
    horiz = power_2d[cy - halfwidth : cy + halfwidth + 1, :].mean()
    vert = power_2d[:, cx - halfwidth : cx + halfwidth + 1].mean()
    return float(horiz / max(vert, 1e-12))


def analyse_one(
    g_path: Path,
    f_path: Path,
    fft_size: int = 256,
) -> dict | None:
    """Compute pixel- and frequency-domain stats for one (g, f) pair.

    Returns None if either file fails to load."""
    try:
        g = _open_rgb01(g_path)
        f = _open_rgb01(f_path)
    except Exception:
        return None
    if g.shape != f.shape:
        f = np.asarray(
            Image.fromarray((f * 255).astype(np.uint8)).resize(
                (g.shape[1], g.shape[0]), Image.BICUBIC
            )
        ) / 255.0
    delta = g - f

    # --- pixel-space stats (full resolution, grayscale) ---
    delta_gray = delta.mean(axis=-1)
    abs_delta = np.abs(delta_gray)
    pix = {
        "linf": float(abs_delta.max()),
        "l2": float(np.sqrt((delta_gray ** 2).mean())),
        "frac_visible": float((abs_delta > 1.0 / 255.0).mean()),
        "spatial_entropy_norm": _spatial_entropy(abs_delta),
        "top_decile_energy_frac": _spatial_top_decile_fraction(abs_delta),
    }

    # --- frequency-space stats at the model's input resolution ---
    g_pool = _avg_pool_to(g, fft_size).mean(axis=-1)
    f_pool = _avg_pool_to(f, fft_size).mean(axis=-1)
    d_pool = g_pool - f_pool

    spec_g = np.fft.fftshift(np.abs(np.fft.fft2(g_pool)))
    spec_f = np.fft.fftshift(np.abs(np.fft.fft2(f_pool)))
    spec_d = np.fft.fftshift(np.abs(np.fft.fft2(d_pool)))

    pow_g = spec_g ** 2
    pow_f = spec_f ** 2
    pow_d = spec_d ** 2

    profile_g = _radial_profile(pow_g)
    profile_f = _radial_profile(pow_f)
    profile_d = _radial_profile(pow_d)

    freq = {
        "delta_linf_pooled": float(np.abs(d_pool).max()),
        "energy_frac_central_32": _central_energy_fraction(pow_d, 32),
        "energy_frac_central_64": _central_energy_fraction(pow_d, 64),
        "energy_frac_central_128": _central_energy_fraction(pow_d, 128),
        "anisotropy_delta": _anisotropy(pow_d),
        "anisotropy_g": _anisotropy(pow_g),
        "log_g_mean": float(np.log1p(spec_g).mean()),
        "log_g_std": float(np.log1p(spec_g).std()),
        "radial_profile_delta": profile_d.tolist(),
        "radial_profile_g": profile_g.tolist(),
        "radial_profile_f": profile_f.tolist(),
    }

    return {"pixel": pix, "frequency": freq}


def run_dataset(
    metadata_path: Path,
    images_dir: Path,
    baseline_dir: Path,
    fft_size: int = 256,
    prompt_ids: Iterable[int] | None = None,
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    metadata = json.loads(metadata_path.read_text())
    rows: list[dict] = []

    for entry in metadata:
        p_idx = _prompt_idx_from_entry(entry)
        if p_idx is None:
            continue
        if prompt_ids is not None and p_idx not in prompt_ids:
            continue
        bits = entry.get("bits")
        g_path = images_dir / entry["file"]
        f_path = _baseline_path(baseline_dir, p_idx)
        if f_path is None or not g_path.exists():
            continue

        result = analyse_one(g_path, f_path, fft_size=fft_size)
        if result is None:
            continue
        result["file"] = entry["file"]
        result["prompt_idx"] = p_idx
        result["bits"] = bits
        rows.append(result)
        if limit is not None and len(rows) >= limit:
            break

    # Aggregate per-bit radial profiles.
    profiles_by_bit: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        bits = row.get("bits") or [0] * 8
        prof = np.asarray(row["frequency"]["radial_profile_delta"])
        for i, b in enumerate(bits):
            profiles_by_bit[(i, int(b))].append(prof)

    aggregate = {}
    for (bit_idx, val), profs in profiles_by_bit.items():
        stack = np.stack(profs)
        aggregate.setdefault(f"bit_{bit_idx}", {})[f"val_{val}_mean"] = stack.mean(axis=0).tolist()
        aggregate.setdefault(f"bit_{bit_idx}", {})[f"val_{val}_n"] = int(stack.shape[0])

    # L2 distance between bit=0 and bit=1 mean radial profiles, per bit.
    for bit_idx in range(8):
        key = f"bit_{bit_idx}"
        if key not in aggregate:
            continue
        m0 = np.asarray(aggregate[key].get("val_0_mean") or [])
        m1 = np.asarray(aggregate[key].get("val_1_mean") or [])
        if m0.size and m1.size:
            n = min(m0.size, m1.size)
            log0 = np.log1p(m0[:n])
            log1 = np.log1p(m1[:n])
            aggregate[key]["log_l2"] = float(np.linalg.norm(log0 - log1))
            denom = np.maximum(0.5 * (m0[:n] + m1[:n]), 1e-12)
            aggregate[key]["rel_l1"] = float(np.mean(np.abs(m1[:n] - m0[:n]) / denom))

    return rows, aggregate


def make_figures(rows: list[dict], aggregate: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mean radial profile (delta vs g vs f).
    profs_d = np.stack([np.asarray(r["frequency"]["radial_profile_delta"]) for r in rows])
    profs_g = np.stack([np.asarray(r["frequency"]["radial_profile_g"]) for r in rows])
    profs_f = np.stack([np.asarray(r["frequency"]["radial_profile_f"]) for r in rows])
    k = np.arange(profs_d.shape[1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(k[1:], profs_g.mean(0)[1:], label="$|F(g)|^2$ (watermarked, mean)")
    ax.loglog(k[1:], profs_f.mean(0)[1:], "--", label="$|F(f)|^2$ (baseline, mean)")
    ax.loglog(k[1:], profs_d.mean(0)[1:], label="$|F(\\Delta)|^2$ (mean)")
    ax.set_xlabel("radial bin $k$")
    ax.set_ylabel("mean power")
    ax.set_title(f"Radial power spectrum, mean over {len(rows)} (prompt, bits) pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_radial_profile.png", dpi=160)
    plt.close(fig)

    # Per-bit radial profiles (bit=1 mean / bit=0 mean ratio).
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for bit_idx in range(8):
        ax = axes[bit_idx // 4, bit_idx % 4]
        key = f"bit_{bit_idx}"
        if key not in aggregate:
            continue
        m0 = np.asarray(aggregate[key].get("val_0_mean") or [])
        m1 = np.asarray(aggregate[key].get("val_1_mean") or [])
        n = min(m0.size, m1.size)
        if n < 2:
            continue
        ratio = m1[:n] / np.maximum(m0[:n], 1e-12)
        kk = np.arange(n)
        ax.semilogx(kk[1:], ratio[1:])
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_title(f"bit {bit_idx}: P(k|bit=1) / P(k|bit=0)")
        ax.set_ylim(0.5, 2.0)
    for ax in axes[-1]:
        ax.set_xlabel("radial bin $k$")
    fig.tight_layout()
    fig.savefig(out_dir / "per_bit_radial_profile.png", dpi=160)
    plt.close(fig)

    # Energy concentration histograms (delta).
    e32 = np.asarray([r["frequency"]["energy_frac_central_32"] for r in rows])
    e64 = np.asarray([r["frequency"]["energy_frac_central_64"] for r in rows])
    e128 = np.asarray([r["frequency"]["energy_frac_central_128"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 60)
    for arr, lab, ref in [
        (e32, "central $32^2$", (2 * 32) ** 2 / (256 ** 2)),
        (e64, "central $64^2$", (2 * 64) ** 2 / (256 ** 2)),
        (e128, "central $128^2$", (2 * 128) ** 2 / (256 ** 2)),
    ]:
        ax.hist(arr, bins=bins, alpha=0.5, label=f"{lab}  (uniform={ref:.2f})")
    ax.set_xlabel("energy fraction of $|F(\\Delta)|^2$ in low-freq box")
    ax.set_ylabel("count")
    ax.set_title("Spectral concentration of $\\Delta$ at the model's 256$^2$ FFT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "energy_concentration_hist.png", dpi=160)
    plt.close(fig)

    # Spatial concentration of |Delta|.
    tdf = np.asarray([r["pixel"]["top_decile_energy_frac"] for r in rows])
    ent = np.asarray([r["pixel"]["spatial_entropy_norm"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(tdf, bins=40)
    axes[0].set_title("top-decile spatial energy fraction of $|\\Delta|$")
    axes[0].axvline(0.1, color="k", linestyle="--", label="uniform = 0.1")
    axes[0].legend()
    axes[1].hist(ent, bins=40)
    axes[1].set_title("normalised spatial entropy of $|\\Delta|$  (1 = uniform)")
    axes[1].axvline(1.0, color="k", linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "spatial_concentration_hist.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--metadata", default=None)
    p.add_argument("--images", default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--fft-size", type=int, default=256,
                   help="Pool to this size before FFT (matches DualBranch.spec_downsample).")
    p.add_argument("--prompt-ids", default=None,
                   help="Comma-separated prompt ids to restrict to (default: all).")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many entries (debug).")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-fig-dir", default=None)
    args = p.parse_args()

    metadata = Path(args.metadata) if args.metadata else REPO / "watermark_encoding" / "data" / "metadata.json"
    images_dir = Path(args.images) if args.images else REPO / "watermark_encoding" / "data" / "images"
    baseline_dir = Path(args.baseline) if args.baseline else REPO / "watermark_encoding" / "data" / "baseline"

    out_json = Path(args.out_json) if args.out_json else DECODING_ROOT / "results" / "delta_dataset.json"
    out_fig = Path(args.out_fig_dir) if args.out_fig_dir else DECODING_ROOT / "results" / "figures" / "delta_dataset"

    prompt_ids = None
    if args.prompt_ids:
        prompt_ids = {int(s) for s in args.prompt_ids.split(",")}

    print(f"[delta_dataset] metadata={metadata}")
    print(f"[delta_dataset] images={images_dir}")
    print(f"[delta_dataset] baseline={baseline_dir}")
    print(f"[delta_dataset] fft_size={args.fft_size}")

    rows, aggregate = run_dataset(
        metadata_path=metadata,
        images_dir=images_dir,
        baseline_dir=baseline_dir,
        fft_size=args.fft_size,
        prompt_ids=prompt_ids,
        limit=args.limit,
    )
    print(f"[delta_dataset] processed {len(rows)} pairs")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows, "aggregate": aggregate, "fft_size": args.fft_size}, indent=2))
    print(f"[delta_dataset] wrote {out_json}")

    make_figures(rows, aggregate, out_fig)
    print(f"[delta_dataset] figures: {out_fig}/")


if __name__ == "__main__":
    main()
