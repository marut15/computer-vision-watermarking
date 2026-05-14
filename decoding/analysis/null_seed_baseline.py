"""Null-distribution control: characterise (baseline_a - baseline_b) for the
same prompt and different seeds.

The puzzle figure shows that Delta = (g - f) is broadband and spatially
localised to texture regions. The natural null is: how much of that is just
``two SDXL samples of the same prompt look about this different''?  If
(baseline_a - baseline_b) for two un-watermarked seeds at the same prompt is
*also* broadband and spatially localised with similar magnitude, then the
``broadband, spatially localised'' description applies to *generation noise*
in general; the bit-relevant signal must live in something else.

This script needs a small dataset of multiple un-watermarked SDXL samples
per prompt, which the original encoding pipeline does not produce by default
(only one ``baseline'' per prompt). It is therefore set up to operate in two
modes:

  1. ``--baseline-pool DIR`` - if you have a folder with multiple baseline
     PNGs per prompt named e.g. ``baseline_p00_seed{0..N}.png``, the script
     enumerates pairs and computes Delta_null statistics analogous to
     decoding/scripts/delta_dataset_analysis.py.

  2. ``--regenerate`` - reuse the encoding pipeline to generate K extra
     baselines per prompt and analyse them. Requires GPU + the encoding
     environment; we don't run this automatically because it is expensive.

The output schema mirrors ``delta_dataset.json`` so the two can be compared
directly with ``compare_delta_vs_null.py`` (TODO if the dataset exists).

Outputs:
  - decoding/results/delta_null.json
  - decoding/results/figures/delta_null_radial.png

Usage:
  python decoding/scripts/null_seed_baseline.py --baseline-pool path/to/pool

Status: this is the only experiment in the suite that requires *new data
generation*; if no extra-seed pool is available, the script will print a
clear error and exit. The accompanying report flags ``null distribution''
as an open question.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent
REPO = DECODING_ROOT.parent

PROMPT_RE = re.compile(r"baseline_p(\d+)(?:_seed(\d+))?\.png$")


def _open_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _radial_profile(power_2d: np.ndarray) -> np.ndarray:
    h, w = power_2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    nbins = min(cy, cx) + 1
    out = np.zeros(nbins)
    counts = np.zeros(nbins, dtype=np.int64)
    flat_r = r.ravel()
    flat_p = power_2d.ravel()
    np.add.at(out, flat_r[flat_r < nbins], flat_p[flat_r < nbins])
    np.add.at(counts, flat_r[flat_r < nbins], 1)
    return out / np.maximum(counts, 1)


def index_pool(pool_dir: Path) -> dict[int, list[Path]]:
    pool: dict[int, list[Path]] = defaultdict(list)
    for f in sorted(pool_dir.glob("baseline_p*.png")):
        m = PROMPT_RE.search(f.name)
        if not m:
            continue
        pool[int(m.group(1))].append(f)
    return pool


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--baseline-pool", required=False, default=None,
                   help="Directory containing ``baseline_p{P:02d}_seed{S}.png`` files.")
    p.add_argument("--limit-pairs-per-prompt", type=int, default=8)
    p.add_argument("--out-json", default=str(DECODING_ROOT / "results" / "delta_null.json"))
    p.add_argument("--out-fig", default=str(DECODING_ROOT / "results" / "figures" / "delta_null_radial.png"))
    return p.parse_args()


def main():
    args = parse_args()
    if not args.baseline_pool:
        raise SystemExit(
            "no --baseline-pool provided; this script needs >=2 baselines per "
            "prompt (different seeds) to compute the null distribution. "
            "Generate a pool with the encoding pipeline (e.g. "
            "encoding/scripts/generate_baseline.py with --num-seeds 8) and "
            "rerun. The report flags this as an open experiment."
        )
    pool_dir = Path(args.baseline_pool)
    pool = index_pool(pool_dir)
    if not pool:
        raise SystemExit(f"no baselines under {pool_dir}")
    print(f"[null] prompts={sorted(pool)} per-prompt counts="
          f"{[len(v) for v in pool.values()]}")

    rows = []
    for prompt_idx, paths in pool.items():
        # Take the first one as the reference; pair the rest against it.
        if len(paths) < 2:
            continue
        ref = _open_rgb01(paths[0])
        for f in paths[1 : 1 + args.limit_pairs_per_prompt]:
            other = _open_rgb01(f)
            if other.shape != ref.shape:
                other = np.asarray(
                    Image.fromarray((other * 255).astype(np.uint8)).resize(
                        (ref.shape[1], ref.shape[0]), Image.BICUBIC
                    )
                ) / 255.0
            delta = (other - ref).mean(axis=-1)
            F = np.fft.fftshift(np.fft.fft2(delta))
            pow_d = np.abs(F) ** 2
            rp = _radial_profile(pow_d)
            rows.append({
                "prompt_idx": prompt_idx,
                "ref_file": paths[0].name,
                "other_file": f.name,
                "linf": float(np.abs(delta).max()),
                "l2": float(np.sqrt((delta ** 2).mean())),
                "energy_frac_central_128": float(
                    pow_d[
                        pow_d.shape[0] // 2 - 128 : pow_d.shape[0] // 2 + 128,
                        pow_d.shape[1] // 2 - 128 : pow_d.shape[1] // 2 + 128,
                    ].sum() / max(pow_d.sum(), 1e-12)
                ),
                "radial_profile": rp.tolist(),
            })

    out = {"n_pairs": len(rows), "rows": rows}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[null] {len(rows)} pairs -> {out_json}")

    if rows:
        profs = np.stack([np.asarray(r["radial_profile"]) for r in rows])
        k = np.arange(profs.shape[1])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.loglog(k[1:], profs.mean(0)[1:], label="null  (baseline_a - baseline_b)")
        ax.set_xlabel("radial bin $k$")
        ax.set_ylabel("mean power")
        ax.set_title(f"Null-distribution radial spectrum  (n={profs.shape[0]} pairs)")
        ax.legend()
        fig.tight_layout()
        out_fig = Path(args.out_fig)
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=160)
        plt.close(fig)
        print(f"[null] figure -> {out_fig}")


if __name__ == "__main__":
    main()
