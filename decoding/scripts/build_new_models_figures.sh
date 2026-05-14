#!/usr/bin/env bash
# Stage training/eval logs for the four new decoders, generate comparison
# figures into <figures-root> for S3 upload, and commit the logs to git so
# they're available for downstream report generation.
#
# Outputs:
#   decoding/results/training_logs/<arch>.log         (training, force-added)
#   decoding/results/training_logs/<arch>.eval.log    (clean eval, force-added)
#   decoding/results/training_logs/<arch>.robust.log  (robustness eval, force-added)
#   <figures-root>/training_loss.png
#   <figures-root>/training_val_curves.png
#   <figures-root>/clean_comparison.png
#   <figures-root>/per_bit_new_models.png
#   <figures-root>/robustness_heatmap.png
#   <figures-root>/dual_branch_vs_resnet.png
#   <figures-root>/figures_manifest.json
#
# Defaults:
#   --staging-root  /workspace/new_models
#   --figures-root  /workspace/new_models_figures
#   --logs-dir      decoding/.train_new/logs
#
# Usage:
#   bash decoding/scripts/build_new_models_figures.sh
#   bash decoding/scripts/build_new_models_figures.sh --no-commit  # figures only
#   bash decoding/scripts/build_new_models_figures.sh --branch main

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECODING="$(cd "${HERE}/.." && pwd)"
REPO="$(cd "${DECODING}/.." && pwd)"

WORKSPACE="${WORKSPACE:-/workspace}"
if [[ -d "${WORKSPACE}" ]]; then
  DEFAULT_STAGING="${WORKSPACE}/new_models"
  DEFAULT_FIGURES="${WORKSPACE}/new_models_figures"
else
  DEFAULT_STAGING="${REPO}/new_models"
  DEFAULT_FIGURES="${REPO}/new_models_figures"
fi

STAGING_ROOT="${STAGING_ROOT:-${DEFAULT_STAGING}}"
FIGURES_ROOT="${FIGURES_ROOT:-${DEFAULT_FIGURES}}"
LOGS_DIR_DEFAULT="${DECODING}/.train_new/logs"
LOGS_DIR="${LOGS_DIR_DEFAULT}"
OLD_RESULTS="${DECODING}/results"
PYTHON="${PYTHON:-python3}"
NO_COMMIT=0
BRANCH=""  # default: current branch

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [options]

Options:
  --staging-root DIR   Per-arch staging dir (default: ${STAGING_ROOT})
  --figures-root DIR   Output dir for figures (default: ${FIGURES_ROOT})
  --logs-dir DIR       Training log dir (default: ${LOGS_DIR_DEFAULT})
  --old-results DIR    Old-model results dir (default: ${OLD_RESULTS})
  --branch NAME        Push to this branch (default: current)
  --no-commit          Generate figures + stage logs in tree, don't commit/push
  -h | --help          Show this help
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --staging-root) STAGING_ROOT="$2"; shift 2 ;;
    --figures-root) FIGURES_ROOT="$2"; shift 2 ;;
    --logs-dir)     LOGS_DIR="$2"; shift 2 ;;
    --old-results)  OLD_RESULTS="$2"; shift 2 ;;
    --branch)       BRANCH="$2"; shift 2 ;;
    --no-commit)    NO_COMMIT=1; shift ;;
    -h|--help)      usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

if [[ ! -d "${LOGS_DIR}" ]]; then
  echo "[build-figs] ERROR: logs dir not found: ${LOGS_DIR}" >&2
  echo "             (run train_new_decoders.sh first)" >&2
  exit 2
fi
if [[ ! -d "${STAGING_ROOT}" ]]; then
  echo "[build-figs] ERROR: staging root not found: ${STAGING_ROOT}" >&2
  echo "             (run eval_new_decoders.sh first)" >&2
  exit 2
fi

LOG_STAGING="${DECODING}/results/training_logs"
mkdir -p "${LOG_STAGING}"
mkdir -p "${FIGURES_ROOT}"

echo "[build-figs] staging:  ${STAGING_ROOT}"
echo "[build-figs] figures:  ${FIGURES_ROOT}"
echo "[build-figs] logs in:  ${LOGS_DIR}"
echo "[build-figs] logs out: ${LOG_STAGING}"

# 1. Stage training logs into the repo (force-add since *.log is gitignored).
declare -a STAGED=()
for src in "${LOGS_DIR}"/global_stats.log \
           "${LOGS_DIR}"/spectral.log \
           "${LOGS_DIR}"/multiscale.log \
           "${LOGS_DIR}"/dual_branch.log; do
  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${LOG_STAGING}/$(basename "${src}")"
    STAGED+=("${LOG_STAGING}/$(basename "${src}")")
    echo "[build-figs]   staged $(basename "${src}")"
  else
    echo "[build-figs]   WARN: missing ${src}"
  fi
done

# Also stage the per-arch eval and robustness logs from /workspace/new_models/.
for arch in global_stats spectral multiscale_pyramid dual_branch; do
  for kind in evaluate robustness; do
    src="${STAGING_ROOT}/${arch}/${kind}.log"
    [[ -f "${src}" ]] || continue
    short_kind="eval"; [[ "${kind}" == "robustness" ]] && short_kind="robust"
    dst="${LOG_STAGING}/${arch}.${short_kind}.log"
    cp -f "${src}" "${dst}"
    STAGED+=("${dst}")
    echo "[build-figs]   staged ${arch}.${short_kind}.log"
  done
done

# 2. Generate figures.
echo "[build-figs] generating figures..."
"${PYTHON}" "${HERE}/../analysis/build_new_models_figures.py" \
  --staging-root "${STAGING_ROOT}" \
  --figures-root "${FIGURES_ROOT}" \
  --logs-dir "${LOGS_DIR}" \
  --old-results-dir "${OLD_RESULTS}"

# 3. Commit logs to git (force-add since *.log is gitignored). The .gitignore
#    has an explicit exception for results/training_logs/*.log so once that
#    lands, future git add no longer needs --force.
if [[ "${NO_COMMIT}" -eq 0 ]]; then
  cd "${REPO}"
  CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  TARGET_BRANCH="${BRANCH:-${CUR_BRANCH}}"
  if [[ "${CUR_BRANCH}" != "${TARGET_BRANCH}" ]]; then
    echo "[build-figs] switching from ${CUR_BRANCH} to ${TARGET_BRANCH}"
    git checkout "${TARGET_BRANCH}"
  fi

  # Force-add since *.log is gitignored repo-wide (the gitignore exception
  # may not be in this checkout yet).
  if [[ "${#STAGED[@]}" -gt 0 ]]; then
    git add -f "${STAGED[@]}"
  fi
  # Also pick up .gitignore changes if any.
  git add "${REPO}/decoding/.gitignore" 2>/dev/null || true

  if git diff --cached --quiet; then
    echo "[build-figs] no log changes to commit"
  else
    git commit -m "Stage new-decoder training + eval logs for analysis

Logs from train_new_decoders.sh and eval_new_decoders.sh, copied into
decoding/results/training_logs/ so they're available in the repo for
the next step (writing performance markdowns from the per-epoch
trajectories)."
    echo "[build-figs] pushing to ${TARGET_BRANCH}..."
    attempt=0
    delay=2
    until git push -u origin "${TARGET_BRANCH}"; do
      attempt=$((attempt + 1))
      if (( attempt >= 4 )); then
        echo "[build-figs] push failed after ${attempt} attempts" >&2
        exit 1
      fi
      echo "[build-figs] push failed; retrying in ${delay}s (attempt ${attempt}/4)"
      sleep "${delay}"
      delay=$((delay * 2))
    done
  fi
else
  echo "[build-figs] --no-commit: skipping git commit/push"
fi

# 4. Summary.
echo
echo "==================================================="
echo "[build-figs] figures (ready for S3) under ${FIGURES_ROOT}:"
echo "==================================================="
find "${FIGURES_ROOT}" -maxdepth 1 -type f | sort | while read -r f; do
  printf "  %-90s (%s)\n" "${f#${FIGURES_ROOT}/}" "$(du -h "${f}" 2>/dev/null | cut -f1)"
done

echo
echo "[build-figs] to upload figures to S3:"
echo "    aws s3 sync ${FIGURES_ROOT}/ s3://<your-bucket>/new_models_figures/"
