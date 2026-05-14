#!/usr/bin/env bash
# Compute Delta = (watermarked - baseline) statistics across the dataset
# WITHOUT any prior setup: this script bootstraps python3 + numpy + pillow
# if they aren't already installed, then runs an inline analysis. No git or
# repo checkout required - just point it at the data directory.
#
# Outputs:
#   - prints summary table to stdout (mean / median / p95 ||Delta||_inf,
#     percentage of pixels above six threshold levels)
#   - writes delta_stats.json with the same numbers, machine-readable
#
# Usage examples:
#   bash measure_delta_stats.sh
#   bash measure_delta_stats.sh --data /workspace/data --num-images 500
#   bash measure_delta_stats.sh \
#     --metadata /path/metadata.json \
#     --images   /path/images \
#     --baseline /path/baseline \
#     --output   delta_stats.json
#
# The script tries common data layouts automatically:
#   <DATA>/metadata.json + <DATA>/images/ + <DATA>/baseline/
# searched under /workspace/computer-vision-watermarking/watermark_encoding/data,
# /workspace/watermark_encoding/data, /workspace/data, $HOME/data, ./data.

set -euo pipefail

# ---------- defaults ----------
META=""
IMAGES=""
BASELINE=""
DATA=""
NUM_IMAGES=200
OUTPUT_JSON="delta_stats.json"

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [options]

Options:
  --data DIR           Root containing metadata.json + images/ + baseline/
  --metadata PATH      Override metadata.json path
  --images DIR         Override watermarked images directory
  --baseline DIR       Override baseline images directory
  --num-images N       How many metadata entries to process (default: ${NUM_IMAGES})
  --output PATH        Output JSON path (default: ${OUTPUT_JSON})
  -h | --help          Show this help
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)        DATA="$2"; shift 2 ;;
    --metadata)    META="$2"; shift 2 ;;
    --images)      IMAGES="$2"; shift 2 ;;
    --baseline)    BASELINE="$2"; shift 2 ;;
    --num-images)  NUM_IMAGES="$2"; shift 2 ;;
    --output)      OUTPUT_JSON="$2"; shift 2 ;;
    -h|--help)     usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

# ---------- resolve paths ----------
if [[ -n "${DATA}" ]]; then
  [[ -z "${META}"     ]] && META="${DATA}/metadata.json"
  [[ -z "${IMAGES}"   ]] && IMAGES="${DATA}/images"
  [[ -z "${BASELINE}" ]] && BASELINE="${DATA}/baseline"
fi

if [[ -z "${META}" ]]; then
  for cand in \
    "/workspace/computer-vision-watermarking/watermark_encoding/data" \
    "/workspace/watermark_encoding/data" \
    "/workspace/data" \
    "${HOME}/data" \
    "./data"; do
    if [[ -f "${cand}/metadata.json" ]]; then
      META="${cand}/metadata.json"
      IMAGES="${cand}/images"
      BASELINE="${cand}/baseline"
      echo "[delta] found data at ${cand}"
      break
    fi
  done
fi

if [[ ! -f "${META}" ]]; then
  echo "[delta] ERROR: cannot find metadata.json" >&2
  echo "        try: bash $(basename "$0") --data /path/to/data" >&2
  exit 2
fi
[[ ! -d "${IMAGES}"   ]] && { echo "[delta] ERROR: images dir not found: ${IMAGES}"     >&2; exit 2; }
[[ ! -d "${BASELINE}" ]] && { echo "[delta] ERROR: baseline dir not found: ${BASELINE}" >&2; exit 2; }

echo "[delta] metadata: ${META}"
echo "[delta] images:   ${IMAGES}"
echo "[delta] baseline: ${BASELINE}"
echo "[delta] num:      ${NUM_IMAGES}"
echo "[delta] output:   ${OUTPUT_JSON}"

# ---------- bootstrap python ----------
if ! command -v python3 >/dev/null 2>&1; then
  echo "[delta] python3 missing; installing"
  if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
      SUDO="sudo"
    fi
    ${SUDO} apt-get update -qq
    ${SUDO} apt-get install -y -qq python3 python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip
  elif command -v brew >/dev/null 2>&1; then
    brew install python
  else
    echo "[delta] ERROR: no apt-get / dnf / brew available; install python3 manually" >&2
    exit 3
  fi
fi

if ! python3 -c "import numpy, PIL" 2>/dev/null; then
  echo "[delta] installing numpy + pillow"
  if ! python3 -m pip install --quiet numpy pillow 2>/dev/null; then
    # PEP 668 (Ubuntu 24+) blocks system pip; retry with override.
    python3 -m pip install --quiet --break-system-packages numpy pillow
  fi
fi

# ---------- run the analysis ----------
python3 - "${META}" "${IMAGES}" "${BASELINE}" "${NUM_IMAGES}" "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

meta_path, images_dir, baseline_dir, num, out_path = sys.argv[1:]
num = int(num)
meta = json.loads(Path(meta_path).read_text())
images_dir = Path(images_dir)
baseline_dir = Path(baseline_dir)


def baseline_for_prompt(p):
    """Match either of the conventions used in the repo's data layout."""
    cands = [
        baseline_dir / f"prompt_{p:02d}" / "baseline.png",
        baseline_dir / f"baseline_p{p:02d}.png",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def prompt_idx(entry):
    if "prompt_idx" in entry:
        return int(entry["prompt_idx"])
    f = entry.get("file", "")
    if "_p" in f:
        digits = ""
        for ch in f.split("_p", 1)[1]:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else None
    return None


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
linfs = []
mean_abs = []
agg = {t: [] for t in THRESHOLDS}
processed = 0
shape_skipped = 0
missing_baseline = 0
missing_image = 0

# Cache baselines so we only load each one once.
baseline_cache = {}

for entry in meta[:num]:
    p = prompt_idx(entry)
    if p is None:
        continue
    if p not in baseline_cache:
        bp = baseline_for_prompt(p)
        if bp is None:
            missing_baseline += 1
            baseline_cache[p] = None
            continue
        baseline_cache[p] = load(bp)
    f = baseline_cache[p]
    if f is None:
        continue
    gp = images_dir / entry["file"]
    if not gp.exists():
        missing_image += 1
        continue
    g = load(gp)
    if g.shape != f.shape:
        shape_skipped += 1
        continue
    delta = np.abs(g - f).max(axis=-1)  # collapse RGB -> max over channels
    linfs.append(float(delta.max()))
    mean_abs.append(float(delta.mean()))
    for t in THRESHOLDS:
        agg[t].append(float((delta >= t).mean() * 100.0))
    processed += 1

if processed == 0:
    print("[delta] ERROR: 0 images processed - check path conventions", file=sys.stderr)
    sys.exit(4)

result = {
    "n_processed": processed,
    "n_shape_skipped": shape_skipped,
    "n_missing_baseline": missing_baseline,
    "n_missing_image": missing_image,
    "linf": {
        "mean": float(np.mean(linfs)),
        "median": float(np.median(linfs)),
        "p95": float(np.percentile(linfs, 95)),
        "max": float(np.max(linfs)),
    },
    "mean_abs_delta": {
        "mean": float(np.mean(mean_abs)),
        "median": float(np.median(mean_abs)),
    },
    "fraction_pixels_above_threshold_pct": {
        f"{t * 100:.1f}%": {
            "mean": float(np.mean(agg[t])),
            "median": float(np.median(agg[t])),
        }
        for t in THRESHOLDS
    },
}

# ---------- pretty print ----------
print()
print("=" * 64)
print(f"  Delta statistics across {processed} watermarked images")
print("=" * 64)
print(f"  mean ||Δ||∞       : {result['linf']['mean']:.4f}")
print(f"  median ||Δ||∞     : {result['linf']['median']:.4f}")
print(f"  95th-pct ||Δ||∞   : {result['linf']['p95']:.4f}")
print(f"  max ||Δ||∞        : {result['linf']['max']:.4f}")
print()
print(f"  mean |Δ| per pixel: {result['mean_abs_delta']['mean']:.5f}")
print(f"  (= average pixel changes by {result['mean_abs_delta']['mean'] * 100:.3f}% of dynamic range)")
print()
print("  fraction of pixels with |Δ| ≥ threshold (mean across images):")
for t in THRESHOLDS:
    m = float(np.mean(agg[t]))
    md = float(np.median(agg[t]))
    print(f"    ≥ {t * 100:>5.1f}% of dynamic range : mean = {m:>6.3f}%   median = {md:>6.3f}%")
print()
if shape_skipped or missing_baseline or missing_image:
    print(f"  skipped: {shape_skipped} shape mismatch, "
          f"{missing_baseline} missing baseline, {missing_image} missing image")

Path(out_path).write_text(json.dumps(result, indent=2))
print(f"  wrote {out_path}")
PY
