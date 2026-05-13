#!/usr/bin/env bash
# Train + fully evaluate the DualBranch_R50 variant: bf16 AMP training,
# cosine LR with warm-up, JPEG augmentation, early stopping, then clean
# eval, branch ablation, and the robustness suite. Stages everything into
# /workspace/new_models/dual_branch_r50/ for S3 upload.
#
# Outputs (relative to OUTROOT/dual_branch_r50/):
#   dual_branch_r50.pth        checkpoint (best by val_exact_match)
#   dual_branch_r50.yaml       config
#   training.log               full trainer stdout/stderr
#   training_history.json      per-epoch trajectory
#   clean_metrics.json         from evaluate.py
#   evaluate.log
#   ablation.json              full / no_spectral / no_spatial accuracies
#   ablation.log
#   robustness.json            6 attacks
#   robustness.log
#   figures/{robustness_per_bit,robustness_jpeg_curve}.png
#
# Resumable: each step is skipped if its output already exists. --force redoes.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECODING="$(cd "${HERE}/.." && pwd)"
REPO="$(cd "${DECODING}/.." && pwd)"

WORKSPACE="${WORKSPACE:-/workspace}"
if [[ -d "${WORKSPACE}" ]]; then
  DEFAULT_OUTROOT="${WORKSPACE}/new_models"
else
  DEFAULT_OUTROOT="${REPO}/new_models"
fi
OUTROOT="${OUTROOT:-${DEFAULT_OUTROOT}}"

CONFIG="${DECODING}/configs/dual_branch_r50.yaml"
CKPT="${DECODING}/checkpoints/dual_branch_r50.pth"
ARCH="dual_branch_r50"
NAME="dual_branch_r50"
# IMG is read from CONFIG below (after we know which yaml is active) so the
# bash and yaml never disagree.
IMG=""

BATCH_SIZE=4         # eval batch size (training uses the config value)
NUM_WORKERS=2
PYTHON="${PYTHON:-python3}"
FORCE=0
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_ABLATE=0
SKIP_ROBUSTNESS=0
META=""
IMAGES=""
SPLITS="${DECODING}/data/splits.json"
MAX_EPOCHS=""

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [options]

Options:
  --outroot DIR        Stage results under DIR (default: ${OUTROOT})
  --config PATH        Training config (default: ${CONFIG})
  --batch-size N       Eval batch size (default: ${BATCH_SIZE})
  --num-workers N      DataLoader workers (default: ${NUM_WORKERS})
  --max-epochs N       Override config training.num_epochs
  --metadata PATH      Override metadata.json
  --images PATH        Override watermarked images dir
  --splits PATH        Override splits.json
  --skip-train         Skip training (use existing checkpoint)
  --skip-eval          Skip clean evaluation
  --skip-ablate        Skip branch ablation
  --skip-robustness    Skip robustness eval
  --force              Re-run every step even if output exists
  -h | --help          Show this help
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outroot)         OUTROOT="$2"; shift 2 ;;
    --config)          CONFIG="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
    --max-epochs)      MAX_EPOCHS="$2"; shift 2 ;;
    --metadata)        META="$2"; shift 2 ;;
    --images)          IMAGES="$2"; shift 2 ;;
    --splits)          SPLITS="$2"; shift 2 ;;
    --skip-train)      SKIP_TRAIN=1; shift ;;
    --skip-eval)       SKIP_EVAL=1; shift ;;
    --skip-ablate)     SKIP_ABLATE=1; shift ;;
    --skip-robustness) SKIP_ROBUSTNESS=1; shift ;;
    --force)           FORCE=1; shift ;;
    -h|--help)         usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

DEFAULT_META="${REPO}/watermark_encoding/data/metadata.json"
[[ -z "${META}" && -f "${DEFAULT_META}" ]] && META="${DEFAULT_META}"
[[ -z "${META}" && -f "${REPO}/encoding/data/metadata.json" ]] && META="${REPO}/encoding/data/metadata.json"
DEFAULT_IMAGES="${REPO}/watermark_encoding/data/images"
[[ -z "${IMAGES}" && -d "${DEFAULT_IMAGES}" ]] && IMAGES="${DEFAULT_IMAGES}"

if [[ ! -f "${META}" ]]; then
  echo "[r50] ERROR: cannot find metadata.json (looked at ${DEFAULT_META})" >&2
  exit 2
fi
if [[ ! -d "${IMAGES}" ]]; then
  echo "[r50] ERROR: cannot find images dir (${DEFAULT_IMAGES})" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "[r50] ERROR: missing config ${CONFIG}" >&2
  exit 2
fi

cd "${DECODING}"
TARGET="${OUTROOT}/${NAME}"
mkdir -p "${TARGET}/figures"

# Pull image_size out of the yaml so the eval steps pass the same resolution
# the model was trained at.
IMG="$("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['data']['image_size'])")"

echo "[r50] decoding:   ${DECODING}"
echo "[r50] config:     ${CONFIG}"
echo "[r50] arch:       ${ARCH}"
echo "[r50] image_size: ${IMG}"
echo "[r50] outroot:    ${TARGET}"
echo "[r50] metadata:   ${META}"
echo "[r50] images:     ${IMAGES}"
echo "[r50] splits:     ${SPLITS}"

cp -f "${CONFIG}" "${TARGET}/$(basename "${CONFIG}")"

run_step() {
  # run_step <label> <output> <skip> <log> <command...>
  local label="$1" output="$2" skip="$3" logfile="$4"; shift 4
  if [[ "${skip}" == "1" ]]; then
    echo "[r50] SKIP ${label} (flag)"; return 0
  fi
  if [[ "${FORCE}" -eq 0 && -e "${output}" ]]; then
    echo "[r50] SKIP ${label} (cached: ${output})"; return 0
  fi
  echo "[r50] RUN  ${label}"
  echo "       cmd: $*"
  echo "       log: ${logfile}"
  local t0 t1 dt; t0="$(date +%s)"
  if "$@" >"${logfile}" 2>&1; then
    t1="$(date +%s)"; dt=$(( t1 - t0 ))
    echo "[r50]      PASSED (${dt}s)"
  else
    t1="$(date +%s)"; dt=$(( t1 - t0 ))
    echo "[r50]      FAILED (${dt}s) — tail of log:"
    tail -n 30 "${logfile}" || true
    return 1
  fi
}

# ---------- 1. train ---------- #
TRAIN_LOG="${TARGET}/training.log"
TRAIN_ARGS=("${PYTHON}" scripts/train_dual_branch_efficient.py
            --config "${CONFIG}" --num-workers "${NUM_WORKERS}")
if [[ -n "${MAX_EPOCHS}" ]]; then
  TRAIN_ARGS+=(--max-epochs "${MAX_EPOCHS}")
fi
run_step "train" "${CKPT}" "${SKIP_TRAIN}" "${TRAIN_LOG}" "${TRAIN_ARGS[@]}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[r50] ERROR: training did not produce ${CKPT}" >&2
  exit 2
fi
cp -f "${CKPT}" "${TARGET}/$(basename "${CKPT}")"

# Persist training history into the staging dir if the trainer wrote one.
HIST_SRC="${DECODING}/results/training_logs/${NAME}.history.json"
[[ -f "${HIST_SRC}" ]] && cp -f "${HIST_SRC}" "${TARGET}/training_history.json"

# Stage the training stdout where build_new_models_figures.py can find it
# (LOG_NAME_BY_ARCH expects .train_new/logs/<arch>.log) - this matches the
# convention used by train_new_decoders.sh for the original four decoders.
STAGED_TRAIN_LOG="${DECODING}/.train_new/logs/${NAME}.log"
mkdir -p "$(dirname "${STAGED_TRAIN_LOG}")"
[[ -f "${TRAIN_LOG}" ]] && cp -f "${TRAIN_LOG}" "${STAGED_TRAIN_LOG}"

# ---------- 2. clean eval ---------- #
CLEAN_OUT="${TARGET}/clean_metrics.json"
CLEAN_LOG="${TARGET}/evaluate.log"
run_step "evaluate" "${CLEAN_OUT}" "${SKIP_EVAL}" "${CLEAN_LOG}" \
  "${PYTHON}" scripts/evaluate.py --config "${CONFIG}"
if [[ "${SKIP_EVAL}" -ne 1 ]]; then
  SRC_JSON="${DECODING}/results/test_results/${NAME}.json"
  [[ -f "${SRC_JSON}" ]] && cp -f "${SRC_JSON}" "${CLEAN_OUT}" || \
    echo "[r50] WARN: ${SRC_JSON} missing"
fi

# ---------- 3. branch ablation ---------- #
ABL_OUT="${TARGET}/ablation.json"
ABL_LOG="${TARGET}/ablation.log"
run_step "ablation" "${ABL_OUT}" "${SKIP_ABLATE}" "${ABL_LOG}" \
  "${PYTHON}" scripts/ablate_dual_branch.py \
    --config "${CONFIG}" \
    --arch "${ARCH}" \
    --checkpoint "${CKPT}" \
    --metadata "${META}" --images "${IMAGES}" --splits "${SPLITS}" \
    --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --output-json "${ABL_OUT}"

# ---------- 4. robustness ---------- #
ROB_OUT="${TARGET}/robustness.json"
ROB_LOG="${TARGET}/robustness.log"
run_step "robustness" "${ROB_OUT}" "${SKIP_ROBUSTNESS}" "${ROB_LOG}" \
  "${PYTHON}" scripts/robustness_eval.py \
    --arch "${ARCH}" --checkpoint "${CKPT}" \
    --metadata "${META}" --images "${IMAGES}" --splits "${SPLITS}" \
    --image-size "${IMG}" \
    --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --output-dir "${TARGET}/figures" \
    --results-json "${ROB_OUT}"

# ---------- summary ---------- #
echo
echo "==================================================="
echo "[r50] staged tree (ready for S3):"
echo "==================================================="
find "${TARGET}" -maxdepth 2 -type f | sort | while read -r f; do
  printf "  %-90s (%s)\n" "${f#${TARGET}/}" "$(du -h "${f}" 2>/dev/null | cut -f1)"
done

# Compact digest.
"${PYTHON}" - "${TARGET}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print()
print(f"[r50] digest:")
clean = root / "clean_metrics.json"
ablation = root / "ablation.json"
rob = root / "robustness.json"
if clean.exists():
    d = json.loads(clean.read_text()).get("test_metrics", {})
    print(f"  clean:       mean_bit={d.get('mean_bit_accuracy')}  exact={d.get('exact_match_rate')}")
if ablation.exists():
    d = json.loads(ablation.read_text()).get("modes", {})
    for mode, m in d.items():
        print(f"  ablation {mode:<13}: mean_bit={m.get('mean_bit_accuracy'):.4f}  exact={m.get('exact_match_rate'):.4f}")
if rob.exists():
    d = json.loads(rob.read_text()).get("results", {})
    for atk, m in d.items():
        print(f"  rob {atk:<16}: mean_bit={m.get('mean_bit_accuracy'):.4f}  exact={m.get('exact_match_rate'):.4f}")
PY

echo
echo "[r50] to upload to S3:"
echo "    aws s3 sync ${TARGET}/ s3://<your-bucket>/new_models/${NAME}/"
