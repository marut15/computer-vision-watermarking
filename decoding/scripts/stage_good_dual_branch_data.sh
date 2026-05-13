#!/usr/bin/env bash
# Stage the R-50 dual_branch artefacts into a single folder called
# good_dual_branch_data/, ready to upload to S3 / submit as the final
# project. Pulls in:
#
#   - the model: /workspace/new_models/dual_branch_r50/   (checkpoint,
#     config, training log + history, clean metrics, ablation,
#     robustness, per-arch figures)
#   - the comparison figures: /workspace/new_models_figures/
#     (8-arch clean comparison, robustness heatmap, training curves,
#      dual_branch_vs_resnet, etc.)
#
# Layout produced (under OUTROOT, default /workspace/good_dual_branch_data):
#   model/                        copy of new_models/dual_branch_r50/
#     dual_branch_r50.pth
#     dual_branch_r50.yaml
#     clean_metrics.json
#     ablation.json
#     robustness.json
#     training.log + training_history.json
#     evaluate.log + ablation.log + robustness.log
#     figures/{robustness_per_bit,robustness_jpeg_curve}.png
#   comparison_figures/           copy of new_models_figures/
#     clean_comparison.png
#     robustness_heatmap.png
#     training_loss.png
#     training_val_curves.png
#     dual_branch_vs_resnet.png
#     per_bit_new_models.png
#     figures_manifest.json
#   README.txt                    one-page digest pulled from the JSONs
#   manifest.json                 list of staged files + sizes
#
# Default is copy (safer); pass --move to relocate the originals.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECODING="$(cd "${HERE}/.." && pwd)"
REPO="$(cd "${DECODING}/.." && pwd)"

WORKSPACE="${WORKSPACE:-/workspace}"
if [[ -d "${WORKSPACE}" ]]; then
  DEFAULT_OUTROOT="${WORKSPACE}/good_dual_branch_data"
  DEFAULT_MODEL_SRC="${WORKSPACE}/new_models/dual_branch_r50"
  DEFAULT_FIGURES_SRC="${WORKSPACE}/new_models_figures"
else
  DEFAULT_OUTROOT="${REPO}/good_dual_branch_data"
  DEFAULT_MODEL_SRC="${REPO}/new_models/dual_branch_r50"
  DEFAULT_FIGURES_SRC="${REPO}/new_models_figures"
fi

OUTROOT="${OUTROOT:-${DEFAULT_OUTROOT}}"
MODEL_SRC="${DEFAULT_MODEL_SRC}"
FIGURES_SRC="${DEFAULT_FIGURES_SRC}"
PYTHON="${PYTHON:-python3}"
MOVE=0
FORCE=0

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [options]

Options:
  --outroot DIR        Stage everything under DIR (default: ${OUTROOT})
  --model-src DIR      Source dir for the model (default: ${MODEL_SRC})
  --figures-src DIR    Source dir for the cross-arch figures (default: ${FIGURES_SRC})
  --move               Move the originals instead of copying (rm -rf source after)
  --force              Overwrite OUTROOT if it already exists
  -h | --help          Show this help
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outroot)      OUTROOT="$2"; shift 2 ;;
    --model-src)    MODEL_SRC="$2"; shift 2 ;;
    --figures-src)  FIGURES_SRC="$2"; shift 2 ;;
    --move)         MOVE=1; shift ;;
    --force)        FORCE=1; shift ;;
    -h|--help)      usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

if [[ ! -d "${MODEL_SRC}" ]]; then
  echo "[stage] ERROR: model source not found: ${MODEL_SRC}" >&2
  echo "        (run decoding/scripts/run_dual_branch_r50.sh first)" >&2
  exit 2
fi
if [[ ! -d "${FIGURES_SRC}" ]]; then
  echo "[stage] ERROR: figures source not found: ${FIGURES_SRC}" >&2
  echo "        (run decoding/scripts/build_new_models_figures.sh first)" >&2
  exit 2
fi

if [[ -d "${OUTROOT}" ]]; then
  if [[ "${FORCE}" -eq 1 ]]; then
    echo "[stage] removing existing ${OUTROOT} (--force)"
    rm -rf "${OUTROOT}"
  else
    echo "[stage] ERROR: ${OUTROOT} already exists; pass --force to overwrite" >&2
    exit 2
  fi
fi
mkdir -p "${OUTROOT}/model" "${OUTROOT}/comparison_figures"

echo "[stage] outroot:      ${OUTROOT}"
echo "[stage] model src:    ${MODEL_SRC}"
echo "[stage] figures src:  ${FIGURES_SRC}"
echo "[stage] mode:         $([ "${MOVE}" -eq 1 ] && echo move || echo copy)"

cp -a "${MODEL_SRC}/." "${OUTROOT}/model/"
cp -a "${FIGURES_SRC}/." "${OUTROOT}/comparison_figures/"

# Build a one-page README pulling the headline numbers out of the JSONs.
"${PYTHON}" - "${OUTROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
model_dir = root / "model"

def safe_load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

clean = safe_load(model_dir / "clean_metrics.json") or {}
abl   = safe_load(model_dir / "ablation.json") or {}
rob   = safe_load(model_dir / "robustness.json") or {}

clean_m = clean.get("test_metrics", clean)
abl_modes = abl.get("modes", {})
rob_results = rob.get("results", {})

def pct(x, dp=2):
    return f"{x*100:.{dp}f}%" if isinstance(x, (int, float)) else "n/a"

lines = []
ap = lines.append
ap("DualBranch R-50 (1024x1024) - final results")
ap("=" * 60)
ap("")
ap("Clean test set:")
ap(f"  mean bit accuracy:  {pct(clean_m.get('mean_bit_accuracy'))}")
ap(f"  exact match (8/8):  {pct(clean_m.get('exact_match_rate'))}")
per_bit = clean_m.get("per_bit_accuracy")
if per_bit:
    ap("  per bit:")
    bit_names = ["warm/cool", "sharp/soft", "grainy/clean", "bright/dark",
                 "contrast", "saturation", "detail", "vintage/modern"]
    for i, (a, name) in enumerate(zip(per_bit, bit_names)):
        ap(f"    bit {i} ({name:>16}): {pct(a)}")
ap("")
ap("Branch ablation (which branch carries the signal):")
for mode in ("full", "no_spectral", "no_spatial"):
    if mode in abl_modes:
        m = abl_modes[mode]
        ap(f"  {mode:<14}: mean={pct(m['mean_bit_accuracy'])}  exact={pct(m['exact_match_rate'])}")
ap("")
ap("Robustness suite:")
for atk in ("clean", "jpeg_q90", "jpeg_q75", "jpeg_q50", "resize_512", "random_crop_75"):
    if atk in rob_results:
        m = rob_results[atk]
        ap(f"  {atk:<16}: mean={pct(m['mean_bit_accuracy'])}  exact={pct(m['exact_match_rate'])}")
ap("")
ap("Files:")
ap("  model/                 checkpoint, config, logs, JSONs, per-arch figures")
ap("  comparison_figures/    cross-arch plots from build_new_models_figures.py")

(root / "README.txt").write_text("\n".join(lines) + "\n")
print(f"[stage] wrote {root / 'README.txt'}")
PY

# Manifest of every staged file with size.
"${PYTHON}" - "${OUTROOT}" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
files = []
for p in sorted(root.rglob("*")):
    if p.is_file() and p.name != "manifest.json":
        files.append({
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
        })
total = sum(f["bytes"] for f in files)
manifest = {"outroot": str(root), "n_files": len(files), "total_bytes": total, "files": files}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"[stage] wrote {root / 'manifest.json'}  ({len(files)} files, {total / 1024 / 1024:.1f} MiB)")
PY

if [[ "${MOVE}" -eq 1 ]]; then
  echo "[stage] --move: removing originals"
  rm -rf "${MODEL_SRC}" "${FIGURES_SRC}"
fi

# Listing.
echo
echo "==================================================="
echo "[stage] staged tree (ready for S3):"
echo "==================================================="
find "${OUTROOT}" -maxdepth 3 -type f | sort | while read -r f; do
  printf "  %-90s (%s)\n" "${f#${OUTROOT}/}" "$(du -h "${f}" 2>/dev/null | cut -f1)"
done

echo
echo "[stage] README:"
sed 's/^/  /' "${OUTROOT}/README.txt"

echo
echo "[stage] to upload to S3:"
echo "    aws s3 sync ${OUTROOT}/ s3://<your-bucket>/good_dual_branch_data/"
echo "    # or tar:"
echo "    tar -C $(dirname "${OUTROOT}") -czf $(basename "${OUTROOT}").tar.gz $(basename "${OUTROOT}")"
