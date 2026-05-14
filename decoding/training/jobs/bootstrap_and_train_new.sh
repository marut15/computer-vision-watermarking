#!/usr/bin/env bash
# Single-shot VM bootstrap: clone or locate the repo under /workspace, merge
# S3 staging via setup_workspace.sh, install deps, then train all four new
# global-pattern decoders by handing off to training/jobs/train_new_decoders.sh.
#
# Designed for a fresh Runpod VM (RTX PRO 6000 Blackwell, 96 GB VRAM) with
# the canonical workspace layout:
#
#   /workspace/
#     ├── computer-vision-watermarking/    git checkout (or this script clones)
#     ├── watermark_encoding/              S3 staging (data/ + models/)
#     └── decoding/                        S3 staging (checkpoints/ + results/)
#
# After setup_workspace.sh runs, the staging folders are merged into the
# repo at  <repo>/watermark_encoding/  and  <repo>/decoding/{checkpoints,results}/,
# matching the relative paths in decoding/configs/*.yaml.
#
# Usage on the VM:
#
#     bash computer-vision-watermarking/decoding/training/jobs/bootstrap_and_train_new.sh
#
# Common knobs:
#   --workspace PATH       Workspace root (default: /workspace)
#   --repo-url URL         Git URL to clone if no repo present
#   --branch NAME          Branch to check out (default: main)
#   --skip-setup           Skip setup_workspace.sh (use if data is already merged)
#   --skip-deps            Skip dep install (use if env is already set up)
#   --skip-train           Stop after setup + deps; don't kick off training
#   --force                Pass --force through to train_new_decoders.sh
#   --smoke                Run the synthetic 64-image smoke pipeline instead
#   --train-args "..."     Extra args forwarded verbatim to train_new_decoders.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/matteici/computer-vision-watermarking.git}"
BRANCH="${BRANCH:-main}"
SKIP_SETUP=0
SKIP_DEPS=0
SKIP_TRAIN=0
FORCE_TRAIN=0
SMOKE=0
EXTRA_TRAIN_ARGS=""

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [options]

Options:
  --workspace PATH      Workspace root (default: ${WORKSPACE})
  --repo-url URL        Git URL to clone if no repo is present
                        (default: ${REPO_URL})
  --branch NAME         Branch to check out (default: ${BRANCH})
  --skip-setup          Skip setup_workspace.sh (data already merged)
  --skip-deps           Skip pip dep install
  --skip-train          Stop after setup + deps, don't run training
  --force               Pass --force to train_new_decoders.sh (retrain all)
  --smoke               Run a 2-epoch CPU-friendly smoke training instead
  --train-args "..."    Extra args forwarded to train_new_decoders.sh
  -h | --help           This help
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)    WORKSPACE="$2"; shift 2 ;;
    --repo-url)     REPO_URL="$2"; shift 2 ;;
    --branch)       BRANCH="$2"; shift 2 ;;
    --skip-setup)   SKIP_SETUP=1; shift ;;
    --skip-deps)    SKIP_DEPS=1; shift ;;
    --skip-train)   SKIP_TRAIN=1; shift ;;
    --force)        FORCE_TRAIN=1; shift ;;
    --smoke)        SMOKE=1; shift ;;
    --train-args)   EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
    -h|--help)      usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"

# ---------- 1. locate or clone the repo ----------
REPO=""
for cand in "${WORKSPACE}"/*/.git; do
  [[ -d "${cand}" ]] || continue
  REPO="$(dirname "${cand}")"
  break
done

if [[ -z "${REPO}" ]]; then
  echo "[bootstrap] no git repo found under ${WORKSPACE} — cloning ${REPO_URL}"
  git clone --branch "${BRANCH}" "${REPO_URL}" "${WORKSPACE}/computer-vision-watermarking"
  REPO="${WORKSPACE}/computer-vision-watermarking"
else
  echo "[bootstrap] found repo: ${REPO}"
  if git -C "${REPO}" rev-parse --abbrev-ref HEAD >/dev/null 2>&1; then
    current_branch="$(git -C "${REPO}" rev-parse --abbrev-ref HEAD)"
    if [[ "${current_branch}" != "${BRANCH}" ]]; then
      echo "[bootstrap] checking out ${BRANCH} (was ${current_branch})"
      git -C "${REPO}" fetch --quiet origin "${BRANCH}" || true
      git -C "${REPO}" checkout "${BRANCH}"
    fi
    git -C "${REPO}" pull --ff-only --quiet origin "${BRANCH}" || \
      echo "[bootstrap] WARN: could not fast-forward ${BRANCH} (continuing)"
  fi
fi

DECODING="${REPO}/decoding"
JOBS="${DECODING}/training/jobs"

# ---------- 2. merge S3 staging into the repo ----------
if [[ ${SKIP_SETUP} -eq 0 && ${SMOKE} -eq 0 ]]; then
  echo "[bootstrap] running setup_workspace.sh"
  bash "${REPO}/setup/setup_workspace.sh" --workspace "${WORKSPACE}"
else
  echo "[bootstrap] SKIP setup_workspace.sh"
fi

# ---------- 3. python deps ----------
if [[ ${SKIP_DEPS} -eq 0 ]]; then
  echo "[bootstrap] checking Python deps"
  if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util, sys
need = ["torch", "torchvision", "yaml", "tqdm", "matplotlib", "sklearn", "PIL", "numpy"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY
  then
    echo "[bootstrap] installing missing deps"
    python3 -m pip install --quiet --upgrade pip
    python3 -m pip install --quiet \
      torch torchvision torchcam pyyaml tqdm matplotlib scikit-learn pillow numpy
  else
    echo "[bootstrap] all deps present"
  fi

  python3 - <<'PY'
import torch
print(f"[bootstrap] torch={torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  device {i}: {p.name}  {p.total_memory/1024**3:.1f} GiB VRAM")
PY
fi

# ---------- 4. preflight before training ----------
if [[ ${SMOKE} -eq 0 ]]; then
  DATA_ROOT="${PROJECT_DATA_ROOT:-${WORKSPACE}/data/computer-vision-watermarking}"
  META="${DATA_ROOT}/watermark_encoding/data/metadata.json"
  IMG_DIR="${DATA_ROOT}/watermark_encoding/data/images"
  if [[ ! -f "${META}" ]]; then
    echo "[bootstrap] ERROR: dataset metadata missing at ${META}" >&2
    echo "[bootstrap]        did setup_workspace.sh succeed? did S3 staging exist?" >&2
    exit 4
  fi
  if [[ ! -d "${IMG_DIR}" ]]; then
    echo "[bootstrap] ERROR: images dir missing at ${IMG_DIR}" >&2
    exit 4
  fi
  n_img="$(find "${IMG_DIR}" -maxdepth 1 -name '*.png' | wc -l)"
  echo "[bootstrap] dataset OK: ${n_img} watermarked images at ${IMG_DIR}"
fi

# ---------- 5. hand off to training ----------
if [[ ${SKIP_TRAIN} -eq 1 ]]; then
  echo "[bootstrap] --skip-train set; setup complete. To train later:"
  echo "    bash ${JOBS}/train_new_decoders.sh"
  exit 0
fi

cd "${DECODING}"
TRAIN_FLAGS=()
[[ ${FORCE_TRAIN} -eq 1 ]] && TRAIN_FLAGS+=(--force)
[[ ${SMOKE}      -eq 1 ]] && TRAIN_FLAGS+=(--smoke)
if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${EXTRA_TRAIN_ARGS} )
  TRAIN_FLAGS+=("${EXTRA[@]}")
fi

echo
echo "==================================================="
echo "[bootstrap] handing off to train_new_decoders.sh"
echo "            ${TRAIN_FLAGS[*]:-(no extra flags)}"
echo "==================================================="
exec bash "${JOBS}/train_new_decoders.sh" "${TRAIN_FLAGS[@]}"
