"""2D cepstrum and autocorrelation of Delta = (g - f).

If the LoRA-slider perturbation has a periodic spatial pattern (which would
be invisible in pixel space but loud in frequency space), the cepstrum
``IFFT(log |F(Delta)|^2)`` will show non-trivial peaks. The radius (in
pixels) of a peak is the period of the corresponding repetition.

This script computes:

  - cepstrum  c(x, y) = real(IFFT(log(|F(Delta)|^2 + eps)))  (zero-mean)
  - 2D autocorrelation a(x, y) = real(IFFT(|F(Delta)|^2)) / N
    (just the inverse FT of the power spectrum)

For each test image we keep:
  - log |F(Delta)|^2 mean over the dataset (centred)
  - cepstrum mean over the dataset (zero-mean and zero-variance)
  - locations of the top-K peaks in the cepstrum (excluding origin), and
    their distance from the origin (the candidate spatial period).

Outputs:
  - decoding/results/cepstrum.json   (top peaks + summary stats)
  - decoding/results/figures/cepstrum_delta.png  (heatmaps)

If the cepstrum has clear off-origin peaks at consistent radii across many
images, the LoRA injects a periodic spatial pattern; the spectral CNN can
then read the bit by reading the location/strength of those peaks in the
spectrum. If it's flat, the perturbation is non-periodic and the CNN must
read an envelope feature.

Usage:
  python decoding/scripts/cepstrum_delta.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent
REPO = DECODING_ROOT.parent


def _open_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _baseline_path(baseline_dir: Path, p: int) -> Path | None:
    for c in (
        baseline_dir / f"prompt_{p:02d}" / "baseline.png",
        baseline_dir / f"baseline_p{p:02d}.png",
    ):
        if c.exists():
            return c
    return None


def _prompt_idx(entry: dict) -> int | None:
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


def cepstrum_2d(delta_gray: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    F = np.fft.fft2(delta_gray)
    log_pow = np.log(np.abs(F) ** 2 + eps)
    c = np.real(np.fft.ifft2(log_pow))
    return np.fft.fftshift(c)


def autocorr_2d(delta_gray: np.ndarray) -> np.ndarray:
    F = np.fft.fft2(delta_gray)
    a = np.real(np.fft.ifft2(np.abs(F) ** 2))
    return np.fft.fftshift(a) / delta_gray.size


def top_peaks(image: np.ndarray, k: int = 16, exclude_radius: int = 4) -> list[dict]:
    h, w = image.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    masked = image.copy()
    masked[rr <= exclude_radius] = -np.inf
    flat = masked.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(-flat[idx])]
    out = []
    for j in idx:
        ry, rx = j // w, j % w
        out.append({
            "y": int(ry - cy),
            "x": int(rx - cx),
            "r": float(np.sqrt((ry - cy) ** 2 + (rx - cx) ** 2)),
            "value": float(image[ry, rx]),
        })
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--metadata", default=None)
    p.add_argument("--images", default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--limit", type=int, default=200,
                   help="Process at most this many entries (cepstrum is O(N log N)).")
    p.add_argument("--out-json", default=str(DECODING_ROOT / "results" / "cepstrum.json"))
    p.add_argument("--out-fig", default=str(DECODING_ROOT / "results" / "figures" / "cepstrum_delta.png"))
    return p.parse_args()


def main():
    args = parse_args()
    metadata = Path(args.metadata) if args.metadata else REPO / "watermark_encoding" / "data" / "metadata.json"
    images_dir = Path(args.images) if args.images else REPO / "watermark_encoding" / "data" / "images"
    baseline_dir = Path(args.baseline) if args.baseline else REPO / "watermark_encoding" / "data" / "baseline"

    metadata = json.loads(metadata.read_text())
    accum_cep = None
    accum_logpow = None
    n = 0
    summary_rows = []

    for entry in metadata:
        if n >= args.limit:
            break
        p = _prompt_idx(entry)
        if p is None:
            continue
        f_path = _baseline_path(baseline_dir, p)
        g_path = images_dir / entry["file"]
        if f_path is None or not g_path.exists():
            continue
        try:
            g = _open_rgb01(g_path)
            f = _open_rgb01(f_path)
        except Exception:
            continue
        if g.shape != f.shape:
            f = np.asarray(
                Image.fromarray((f * 255).astype(np.uint8)).resize(
                    (g.shape[1], g.shape[0]), Image.BICUBIC
                )
            ) / 255.0
        d = (g - f).mean(axis=-1)
        F = np.fft.fft2(d)
        logpow = np.log(np.abs(F) ** 2 + 1e-8)
        cep = np.real(np.fft.ifft2(logpow))
        cep_shift = np.fft.fftshift(cep)
        logpow_shift = np.fft.fftshift(logpow)

        if accum_cep is None:
            accum_cep = np.zeros_like(cep_shift)
            accum_logpow = np.zeros_like(logpow_shift)
        accum_cep += cep_shift
        accum_logpow += logpow_shift
        n += 1

        peaks = top_peaks(cep_shift, k=8, exclude_radius=8)
        summary_rows.append({
            "file": entry["file"],
            "prompt_idx": p,
            "bits": entry.get("bits"),
            "linf": float(np.abs(d).max()),
            "top_peaks": peaks,
        })

    if n == 0:
        print("[cepstrum] no usable pairs; aborting")
        return

    mean_cep = accum_cep / n
    mean_logpow = accum_logpow / n

    out = {
        "n": n,
        "mean_top_peaks": top_peaks(mean_cep, k=16, exclude_radius=8),
        "rows_subset": summary_rows[:50],
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[cepstrum] processed {n} pairs; results -> {out_json}")

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(mean_logpow, cmap="magma")
    axes[0].set_title("mean $\\log\\,|F(\\Delta)|^2$")
    axes[0].axis("off")
    cep_view = mean_cep.copy()
    h, w = cep_view.shape
    cy, cx = h // 2, w // 2
    cep_view[cy - 4 : cy + 4, cx - 4 : cx + 4] = np.nan
    vmax = np.nanpercentile(np.abs(cep_view), 99.5)
    axes[1].imshow(cep_view, cmap="seismic", vmin=-vmax, vmax=vmax)
    axes[1].set_title("mean cepstrum (origin masked)")
    axes[1].axis("off")
    fig.suptitle(f"Cepstrum / power-spectrum mean over {n} (g, f) pairs")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)
    print(f"[cepstrum] figure -> {out_fig}")


if __name__ == "__main__":
    main()
