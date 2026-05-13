#!/usr/bin/env bash
# Run all evaluations on the four new decoder architectures
# (global_stats, spectral, multiscale_pyramid, dual_branch) and stage
# everything (checkpoint + config + clean metrics + robustness JSON +
# figures + logs) under <OUTROOT>/<arch>/ ready to upload to S3.
#
# Layout produced (under OUTROOT, default /workspace/new_models):
#   <arch>/<arch>.pth                       checkpoint copied verbatim
#   <arch>/<config>.yaml                    training config copied verbatim
#   <arch>/clean_metrics.json               from evaluate.py (test-set)
#   <arch>/robustness.json                  from robustness_eval.py
#   <arch>/figures/robustness_per_bit.png
#   <arch>/figures/robustness_jpeg_curve.png
#   <arch>/evaluate.log
#   <arch>/robustness.log
#   summary.json                            roll-up across all four
#   manifest.json                           per-step status / duration
#
# Usage:
#   bash decoding/scripts/eval_new_decoders.sh
#   bash decoding/scripts/eval_new_decoders.sh --outroot /tmp/new_models
#   bash decoding/scripts/eval_new_decoders.sh --batch-size 8 --skip-robustness
#   OUTROOT=/path WORKSPACE=/wherever bash decoding/scripts/eval_new_decoders.sh
#
# Resumable: each step is skipped if its output exists. Pass --force to redo.

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

BATCH_SIZE=4
NUM_WORKERS=2
PYTHON="${PYTHON:-python3}"
FORCE=0
SKIP_CLEAN=0
SKIP_ROBUSTNESS=0
META=""
IMAGES=""
SPLITS="${DECODING}/data/splits.json"

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [options]

Options:
  --outroot DIR        Stage results under DIR (default: ${OUTROOT})
  --batch-size N       Batch size for robustness eval (default: ${BATCH_SIZE})
  --num-workers N      DataLoader workers (default: ${NUM_WORKERS})
  --metadata PATH      Override metadata.json
  --images PATH        Override watermarked images dir
  --splits PATH        Override splits.json
  --skip-clean         Skip clean test-set evaluation
  --skip-robustness    Skip robustness eval
  --force              Re-run every step even if output exists
  -h | --help          Show this help
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outroot)         OUTROOT="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
    --metadata)        META="$2"; shift 2 ;;
    --images)          IMAGES="$2"; shift 2 ;;
    --splits)          SPLITS="$2"; shift 2 ;;
    --skip-clean)      SKIP_CLEAN=1; shift ;;
    --skip-robustness) SKIP_ROBUSTNESS=1; shift ;;
    --force)           FORCE=1; shift ;;
    -h|--help)         usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

# Resolve canonical data paths the same way run_full_evaluation.sh does.
DEFAULT_META="${REPO}/watermark_encoding/data/metadata.json"
[[ -z "${META}" && -f "${DEFAULT_META}" ]] && META="${DEFAULT_META}"
[[ -z "${META}" && -f "${REPO}/encoding/data/metadata.json" ]] && META="${REPO}/encoding/data/metadata.json"
DEFAULT_IMAGES="${REPO}/watermark_encoding/data/images"
[[ -z "${IMAGES}" && -d "${DEFAULT_IMAGES}" ]] && IMAGES="${DEFAULT_IMAGES}"

if [[ ! -f "${META}" ]]; then
  echo "[eval-new] ERROR: cannot find metadata.json (looked at ${DEFAULT_META})" >&2
  exit 2
fi
if [[ ! -d "${IMAGES}" ]]; then
  echo "[eval-new] ERROR: cannot find images dir (${DEFAULT_IMAGES})" >&2
  exit 2
fi
if [[ ! -f "${SPLITS}" ]]; then
  echo "[eval-new] ERROR: cannot find splits.json (${SPLITS})" >&2
  exit 2
fi

cd "${DECODING}"
mkdir -p "${OUTROOT}"
MANIFEST="${OUTROOT}/manifest.json"

echo "[eval-new] repo:     ${REPO}"
echo "[eval-new] decoding: ${DECODING}"
echo "[eval-new] outroot:  ${OUTROOT}"
echo "[eval-new] metadata: ${META}"
echo "[eval-new] images:   ${IMAGES}"
echo "[eval-new] splits:   ${SPLITS}"

# arch | yaml config (relative to decoding/) | checkpoint (relative) | image-size for robustness
SPECS=(
  "global_stats|configs/global_stats.yaml|checkpoints/global_stats.pth|256"
  "spectral|configs/spectral.yaml|checkpoints/spectral.pth|1024"
  "multiscale_pyramid|configs/multiscale.yaml|checkpoints/multiscale_pyramid.pth|512"
  "dual_branch|configs/dual_branch.yaml|checkpoints/dual_branch.pth|512"
)

declare -a MANIFEST_ROWS=()
overall_status=0

record() {
  # record <arch> <step> <status> <duration> <output>
  MANIFEST_ROWS+=("    { \"arch\": \"$1\", \"step\": \"$2\", \"status\": \"$3\", \"duration_seconds\": $4, \"output\": \"$5\" }")
}

run_step() {
  # run_step <label> <output_path> <skip_flag> <log_path> <command...>
  local label="$1" output="$2" skip="$3" logfile="$4"; shift 4
  if [[ "${skip}" == "1" ]]; then
    echo "[eval-new] SKIP ${label} (flag)"
    record "${CUR_ARCH}" "${label}" "skipped" 0 "${output}"
    return 0
  fi
  if [[ "${FORCE}" -eq 0 && -e "${output}" ]]; then
    echo "[eval-new] SKIP ${label} (cached: ${output})"
    record "${CUR_ARCH}" "${label}" "cached" 0 "${output}"
    return 0
  fi
  echo "[eval-new] RUN  ${label}"
  echo "           cmd: $*"
  echo "           log: ${logfile}"
  local t0 t1 dt
  t0="$(date +%s)"
  if "$@" >"${logfile}" 2>&1; then
    t1="$(date +%s)"; dt=$(( t1 - t0 ))
    echo "[eval-new]      PASSED (${dt}s)"
    record "${CUR_ARCH}" "${label}" "passed" "${dt}" "${output}"
  else
    t1="$(date +%s)"; dt=$(( t1 - t0 ))
    echo "[eval-new]      FAILED (${dt}s) — tail of log:"
    tail -n 30 "${logfile}" || true
    record "${CUR_ARCH}" "${label}" "failed" "${dt}" "${output}"
    overall_status=1
  fi
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r ARCH CONFIG CKPT IMG <<<"${spec}"
  CUR_ARCH="${ARCH}"

  echo
  echo "==================================================="
  echo "[eval-new] architecture: ${ARCH}"
  echo "           config:       ${CONFIG}"
  echo "           checkpoint:   ${CKPT}"
  echo "           image-size:   ${IMG}"
  echo "==================================================="

  if [[ ! -f "${CKPT}" ]]; then
    echo "[eval-new] ERROR: missing checkpoint ${CKPT} — train it first via train_new_decoders.sh" >&2
    record "${ARCH}" "preflight" "failed" 0 "${CKPT}"
    overall_status=1
    continue
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "[eval-new] ERROR: missing config ${CONFIG}" >&2
    record "${ARCH}" "preflight" "failed" 0 "${CONFIG}"
    overall_status=1
    continue
  fi

  TARGET="${OUTROOT}/${ARCH}"
  mkdir -p "${TARGET}/figures"

  # Stage checkpoint + config (cheap; do it up front so a failed eval still
  # leaves the model behind ready for re-eval).
  cp -f "${CKPT}" "${TARGET}/$(basename "${CKPT}")"
  cp -f "${CONFIG}" "${TARGET}/$(basename "${CONFIG}")"

  # 1) clean evaluation via configs/<arch>.yaml — evaluate.py writes to
  #    decoding/results/test_results/<experiment.name>.json. We copy that to
  #    <target>/clean_metrics.json so the stage dir is self-contained.
  CLEAN_OUT="${TARGET}/clean_metrics.json"
  CLEAN_LOG="${TARGET}/evaluate.log"
  run_step "evaluate" "${CLEAN_OUT}" "${SKIP_CLEAN}" "${CLEAN_LOG}" \
    "${PYTHON}" scripts/evaluate.py --config "${CONFIG}"
  # evaluate.py writes results/test_results/<experiment.name>.json. Experiment
  # name is the YAML's experiment.name field — for multiscale.yaml that's
  # 'multiscale_pyramid', so it matches ${ARCH} for all four configs.
  if [[ "${SKIP_CLEAN}" -ne 1 ]]; then
    SRC_JSON="${DECODING}/results/test_results/${ARCH}.json"
    if [[ -f "${SRC_JSON}" ]]; then
      cp -f "${SRC_JSON}" "${CLEAN_OUT}"
    else
      echo "[eval-new] WARN: expected ${SRC_JSON} not found; clean_metrics.json will be missing"
    fi
  fi

  # 2) robustness via the new --arch / --checkpoint path on robustness_eval.py.
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
done

# ---------- manifest ----------
{
  printf '{\n'
  printf '  "generated_at": "%s",\n' "$(date -u +%FT%TZ)"
  printf '  "outroot": "%s",\n' "${OUTROOT}"
  printf '  "overall_status": %d,\n' "${overall_status}"
  printf '  "rows": [\n'
  n=${#MANIFEST_ROWS[@]}
  for ((i=0; i<n; i++)); do
    if (( i < n - 1 )); then
      printf '%s,\n' "${MANIFEST_ROWS[i]}"
    else
      printf '%s\n' "${MANIFEST_ROWS[i]}"
    fi
  done
  printf '  ]\n}\n'
} > "${MANIFEST}"

# ---------- summary ----------
"${PYTHON}" - "${OUTROOT}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    entry = {}
    clean = d / "clean_metrics.json"
    rob = d / "robustness.json"
    if clean.exists():
        try:
            data = json.loads(clean.read_text())
            entry["clean"] = data.get("test_metrics", data)
            entry["val_exact_match_at_save"] = data.get("val_exact_match")
            entry["checkpoint_epoch"] = data.get("checkpoint_epoch")
        except Exception as e:
            entry["clean_error"] = str(e)
    if rob.exists():
        try:
            data = json.loads(rob.read_text())
            entry["robustness"] = data.get("results", data)
        except Exception as e:
            entry["robustness_error"] = str(e)
    summary[d.name] = entry

out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\n[eval-new] wrote {out}")

# Compact human-readable digest.
print("\n[eval-new] digest:")
for arch, e in summary.items():
    clean = e.get("clean", {})
    rob = e.get("robustness", {})
    cm = clean.get("mean_bit_accuracy")
    ce = clean.get("exact_match_rate")
    line = f"  {arch:>20}: clean mean_bit={cm}  exact={ce}"
    if "clean" in rob:
        rc = rob["clean"]
        line += f"  rob_clean_mean={rc.get('mean_bit_accuracy')}"
    for atk in ("jpeg_q75", "resize_512", "random_crop_75"):
        if atk in rob:
            line += f"  {atk}={rob[atk].get('mean_bit_accuracy')}"
    print(line)
PY

# ---------- listing ----------
echo
echo "==================================================="
echo "[eval-new] staged tree (ready for S3):"
echo "==================================================="
find "${OUTROOT}" -maxdepth 3 -type f \( -name '*.json' -o -name '*.pth' -o -name '*.yaml' -o -name '*.png' -o -name '*.log' \) | sort | while read -r f; do
  printf "  %-90s (%s)\n" "${f#${OUTROOT}/}" "$(du -h "${f}" 2>/dev/null | cut -f1)"
done

echo
if [[ "${overall_status}" -eq 0 ]]; then
  echo "[eval-new] ALL STEPS PASSED OR CACHED"
else
  echo "[eval-new] ONE OR MORE STEPS FAILED — see logs under ${OUTROOT}/<arch>/"
fi

echo
echo "[eval-new] to upload to S3:"
echo "    aws s3 sync ${OUTROOT}/ s3://<your-bucket>/new_models/"
echo "    # or tar:"
echo "    tar -C $(dirname "${OUTROOT}") -czf $(basename "${OUTROOT}").tar.gz $(basename "${OUTROOT}")"

exit "${overall_status}"
