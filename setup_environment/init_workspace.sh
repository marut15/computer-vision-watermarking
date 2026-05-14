#!/usr/bin/env bash
# init_workspace.sh — bootstrap a fresh Runpod / cloud GPU machine.
#
# What it does:
#   1. Clone the repo to /workspace/repo/computer-vision-watermarking
#   2. Sync data from S3 to /workspace/data/computer-vision-watermarking
#   3. pip install all Python dependencies (requirements.txt)
#   4. Write PROJECT_DATA_ROOT to ~/.bashrc and export it now
#   5. Quick sanity check (file counts + project_paths smoke)
#
# Usage:
#   bash init_workspace.sh [--skip-s3] [--skip-deps] [--skip-verify]
#
# Environment:
#   REPO_URL        git remote (default: https://github.com/matteici/computer-vision-watermarking)
#   S3_URI          S3 root uri (default: hard-coded bucket below)
#   WORKSPACE       workspace root (default: /workspace)

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/matteici/computer-vision-watermarking}"
S3_URI="${S3_URI:-s3://watermark-decoder-mai-bocconi-2026-427222695152-us-east-1-an/computer-vision-watermarking}"
REPO_DIR="${WORKSPACE}/repo/computer-vision-watermarking"
DATA_DIR="${WORKSPACE}/data/computer-vision-watermarking"

SKIP_S3=0
SKIP_DEPS=0
SKIP_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-s3)      SKIP_S3=1;      shift ;;
    --skip-deps)    SKIP_DEPS=1;    shift ;;
    --skip-verify)  SKIP_VERIFY=1;  shift ;;
    -h|--help)
      grep "^#" "$0" | head -20 | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

log() { echo "[init] $*"; }

# ── 1. clone repo ────────────────────────────────────────────────────────────
log "=== 1/5  repo ==="
mkdir -p "${WORKSPACE}/repo"
if [[ -d "${REPO_DIR}/.git" ]]; then
  log "repo already present at ${REPO_DIR} — pulling"
  git -C "${REPO_DIR}" pull --ff-only
else
  log "cloning ${REPO_URL} -> ${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

# ── 2. sync from S3 ──────────────────────────────────────────────────────────
if [[ ${SKIP_S3} -eq 0 ]]; then
  log "=== 2/5  s3 sync ==="
  if ! command -v aws &>/dev/null; then
    log "aws cli not found — installing ..."
    pip install --quiet awscli
  fi
  mkdir -p "${DATA_DIR}"
  log "aws s3 sync ${S3_URI}/ -> ${DATA_DIR}/"
  aws s3 sync "${S3_URI}/" "${DATA_DIR}/" \
    --no-progress \
    --exclude "*.DS_Store"
  log "s3 sync complete"
else
  log "=== 2/5  s3 sync SKIPPED ==="
fi

# ── 3. Python dependencies ───────────────────────────────────────────────────
if [[ ${SKIP_DEPS} -eq 0 ]]; then
  log "=== 3/5  python deps ==="
  REQS="${REPO_DIR}/requirements.txt"
  if [[ -f "${REQS}" ]]; then
    pip install --quiet -r "${REQS}"
    log "installed from requirements.txt"
  else
    log "requirements.txt not found — installing known deps directly"
    pip install --quiet \
      torch torchvision numpy Pillow tqdm pyyaml \
      scikit-learn matplotlib scipy \
      diffusers transformers accelerate safetensors \
      boto3 grad-cam
  fi
else
  log "=== 3/5  python deps SKIPPED ==="
fi

# ── 4. PROJECT_DATA_ROOT ──────────────────────────────────────────────────────
log "=== 4/5  environment ==="
export PROJECT_DATA_ROOT="${DATA_DIR}"
PROFILE="${HOME}/.bashrc"

if ! grep -q "PROJECT_DATA_ROOT" "${PROFILE}" 2>/dev/null; then
  echo "" >> "${PROFILE}"
  echo "# computer-vision-watermarking data root" >> "${PROFILE}"
  echo "export PROJECT_DATA_ROOT=\"${DATA_DIR}\"" >> "${PROFILE}"
  log "added PROJECT_DATA_ROOT to ${PROFILE}"
else
  log "PROJECT_DATA_ROOT already in ${PROFILE}"
fi
log "PROJECT_DATA_ROOT=${PROJECT_DATA_ROOT}"

# ── 5. sanity check ───────────────────────────────────────────────────────────
if [[ ${SKIP_VERIFY} -eq 0 ]]; then
  log "=== 5/5  sanity check ==="

  n_img=$(find "${DATA_DIR}/watermark_encoding/data/images" -name "*.png" 2>/dev/null | wc -l || echo 0)
  n_pth=$(find "${DATA_DIR}/decoding/model_bundles" -name "*.pth" 2>/dev/null | wc -l || echo 0)
  n_sft=$(find "${DATA_DIR}" -name "*.safetensors" 2>/dev/null | wc -l || echo 0)
  has_meta=$([[ -f "${DATA_DIR}/watermark_encoding/data/metadata.json" ]] && echo yes || echo MISSING)

  echo "  watermarked images  : ${n_img}  (expect 2560)"
  echo "  decoder .pth        : ${n_pth}  (expect ≥5)"
  echo "  encoder safetensors : ${n_sft}  (expect 16)"
  echo "  metadata.json       : ${has_meta}"

  cd "${REPO_DIR}"
  python3 -c "
import sys
sys.path.insert(0, '.')
from project_paths import Paths
p = Paths()
print(f'  data_root resolved  : {p.data_root}')
print(f'  model_bundles       : {p.model_bundles}')
assert p.data_root.exists(), 'data_root does not exist!'
print('  project_paths OK')
"
else
  log "=== 5/5  sanity check SKIPPED ==="
fi

log ""
log "Workspace ready."
log "  repo : ${REPO_DIR}"
log "  data : ${DATA_DIR}"
log ""
log "Next steps:"
log "  cd ${REPO_DIR}"
log "  python -m decoding.cli test --models all --dry-run --output /tmp/test_out"
log "  python -m decoding.cli decode --model dual_branch_r50 --input <images> --output /tmp/decode_out"
