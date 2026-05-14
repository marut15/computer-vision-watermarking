#!/usr/bin/env bash
# Train each of the four new global-pattern decoders sequentially via
# scripts/train.py and the YAML configs under decoding/configs/.
#
# Outputs (relative to decoding/):
#   checkpoints/global_stats.pth
#   checkpoints/spectral.pth
#   checkpoints/multiscale_pyramid.pth
#   checkpoints/dual_branch.pth
#   .train_new/logs/<model>.log    (one log per model)
#   .train_new/manifest.json       (start/end/duration per step)
#
# Each step is skipped if its checkpoint already exists; pass --force to
# retrain everything. Individual models can be skipped with the --skip-* flags.
#
# Usage:
#   bash decoding/scripts/train_new_decoders.sh
#   bash decoding/scripts/train_new_decoders.sh --force
#   bash decoding/scripts/train_new_decoders.sh --skip-global-stats --skip-spectral
#   bash decoding/scripts/train_new_decoders.sh --smoke      # 2-epoch run on the synthetic fixture

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECODING="$(cd "${HERE}/.." && pwd)"
REPO="$(cd "${DECODING}/.." && pwd)"

PYTHON="${PYTHON:-python3}"
FORCE=0
SMOKE=0
SKIP_GLOBAL_STATS=0
SKIP_SPECTRAL=0
SKIP_MULTISCALE=0
SKIP_DUAL_BRANCH=0

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [options]

Trains all four new global-pattern decoders by invoking scripts/train.py
with the matching YAML config. Each model writes a checkpoint into
decoding/checkpoints/ and a per-model log into decoding/.train_new/logs/.

Options:
  --force               Re-train every model even if a checkpoint exists.
  --smoke               2-epoch run on the synthetic .smoke/ fixture (CPU OK).
  --skip-global-stats   Skip global_stats decoder.
  --skip-spectral       Skip spectral decoder.
  --skip-multiscale     Skip multiscale_pyramid decoder.
  --skip-dual-branch    Skip dual_branch decoder.
  -h | --help           This help.
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)              FORCE=1; shift ;;
    --smoke)              SMOKE=1; shift ;;
    --skip-global-stats)  SKIP_GLOBAL_STATS=1; shift ;;
    --skip-spectral)      SKIP_SPECTRAL=1; shift ;;
    --skip-multiscale)    SKIP_MULTISCALE=1; shift ;;
    --skip-dual-branch)   SKIP_DUAL_BRANCH=1; shift ;;
    -h|--help)            usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

cd "${DECODING}"
LOG_DIR="${DECODING}/.train_new/logs"
MANIFEST="${DECODING}/.train_new/manifest.json"
mkdir -p "${LOG_DIR}" "${DECODING}/checkpoints" "${DECODING}/results"

# In smoke mode we synthesize a tiny dataset and override the configs so that
# train.py points at it. The override configs live under .train_new/configs/.
if [[ "${SMOKE}" -eq 1 ]]; then
  echo "[train] smoke mode — synthesizing fixture and writing override configs"
  "${PYTHON}" _smoke_utils.py
  SMOKE_ROOT="${DECODING}/.smoke"
  SMOKE_META="${SMOKE_ROOT}/metadata.json"
  SMOKE_IMG="${SMOKE_ROOT}/images"
  SMOKE_SPLITS="${SMOKE_ROOT}/splits.json"
  OVERRIDE_DIR="${DECODING}/.train_new/configs"
  SMOKE_CKPT_DIR="${DECODING}/.train_new/smoke_checkpoints"
  mkdir -p "${OVERRIDE_DIR}" "${SMOKE_CKPT_DIR}"
  for arch in global_stats spectral multiscale dual_branch; do
    "${PYTHON}" - "${DECODING}/configs/${arch}.yaml" "${OVERRIDE_DIR}/${arch}.yaml" \
                  "${SMOKE_META}" "${SMOKE_IMG}" "${SMOKE_SPLITS}" "${SMOKE_CKPT_DIR}" <<'PY'
import sys, yaml, os
src, dst, meta, imgs, splits, ckpt_dir = sys.argv[1:7]
cfg = yaml.safe_load(open(src))
cfg["data"]["metadata_path"] = meta
cfg["data"]["images_path"] = imgs
cfg["data"]["splits_path"] = splits
cfg["data"]["image_size"] = 64
cfg["training"]["num_epochs"] = 2
cfg["training"]["batch_size"] = 4
cfg["model"]["pretrained"] = False  # offline / no ImageNet download in smoke mode
ckpt_name = os.path.basename(cfg["output"]["checkpoint"])
cfg["output"]["checkpoint"] = os.path.join(ckpt_dir, ckpt_name)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY
  done
  CONFIG_DIR="${OVERRIDE_DIR}"
else
  CONFIG_DIR="${DECODING}/configs"
fi

declare -a STEPS_RUN=()
declare -a STEPS_SKIPPED=()
overall_status=0
MANIFEST_SEP=""

start_manifest() {
  printf '{\n  "started_at": "%s",\n  "smoke": %s,\n  "steps": [\n' \
    "$(date -u +%FT%TZ)" "$( [[ ${SMOKE} -eq 1 ]] && echo true || echo false )" > "${MANIFEST}"
}

append_manifest() {
  local label="$1" status="$2" duration="$3" output="$4" sep="$5"
  printf '    %s{ "step": "%s", "status": "%s", "duration_seconds": %s, "output": "%s" }\n' \
    "${sep}" "${label}" "${status}" "${duration}" "${output}" >> "${MANIFEST}"
}

close_manifest() {
  printf '  ],\n  "ended_at": "%s",\n  "overall_status": %d\n}\n' \
    "$(date -u +%FT%TZ)" "${overall_status}" >> "${MANIFEST}"
}

run_step() {
  # run_step <label> <skip_flag> <config_yaml>
  local label="$1" skip_flag="$2" config="$3"
  local logfile="${LOG_DIR}/${label}.log"
  # Pull the checkpoint path out of the YAML so we can detect cached runs.
  local output
  output="$("${PYTHON}" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output']['checkpoint'])" "${config}")"
  case "${output}" in
    /*) : ;;
    *)  output="${DECODING}/${output}" ;;
  esac

  if [[ "${skip_flag}" == "1" ]]; then
    echo "[train] SKIP ${label} (--skip-${label})"
    STEPS_SKIPPED+=("${label}")
    append_manifest "${label}" "skipped" 0 "${output}" "${MANIFEST_SEP}"
    MANIFEST_SEP=","
    return 0
  fi
  if [[ "${FORCE}" -eq 0 && -e "${output}" ]]; then
    echo "[train] SKIP ${label} (checkpoint exists: ${output} — pass --force to retrain)"
    STEPS_SKIPPED+=("${label}")
    append_manifest "${label}" "cached" 0 "${output}" "${MANIFEST_SEP}"
    MANIFEST_SEP=","
    return 0
  fi

  echo
  echo "==================================================="
  echo "[train] STEP ${label}"
  echo "        config: ${config}"
  echo "        log:    ${logfile}"
  echo "==================================================="
  local t0 t1 dt
  t0="$(date +%s)"
  if "${PYTHON}" training/train.py --config "${config}" >"${logfile}" 2>&1; then
    t1="$(date +%s)"; dt=$(( t1 - t0 ))
    echo "[train] ${label}: PASSED (${dt}s)  →  ${output}"
    STEPS_RUN+=("${label}")
    append_manifest "${label}" "passed" "${dt}" "${output}" "${MANIFEST_SEP}"
  else
    t1="$(date +%s)"; dt=$(( t1 - t0 ))
    echo "[train] ${label}: FAILED (${dt}s) — see ${logfile}"
    tail -n 25 "${logfile}" || true
    overall_status=1
    append_manifest "${label}" "failed" "${dt}" "${output}" "${MANIFEST_SEP}"
  fi
  MANIFEST_SEP=","
}

start_manifest

run_step "global_stats"       "${SKIP_GLOBAL_STATS}" "${CONFIG_DIR}/global_stats.yaml"
run_step "spectral"           "${SKIP_SPECTRAL}"     "${CONFIG_DIR}/spectral.yaml"
run_step "multiscale"         "${SKIP_MULTISCALE}"   "${CONFIG_DIR}/multiscale.yaml"
run_step "dual_branch"        "${SKIP_DUAL_BRANCH}"  "${CONFIG_DIR}/dual_branch.yaml"

close_manifest

echo
echo "==================================================="
echo "[train] SUMMARY"
echo "==================================================="
echo "ran:      ${STEPS_RUN[*]:-(none)}"
echo "skipped:  ${STEPS_SKIPPED[*]:-(none)}"
echo "manifest: ${MANIFEST}"
echo

if [[ "${SMOKE}" -eq 1 ]]; then
  CKPT_BASE="${DECODING}/.train_new/smoke_checkpoints"
else
  CKPT_BASE="${DECODING}/checkpoints"
fi
for f in \
    "${CKPT_BASE}/global_stats.pth" \
    "${CKPT_BASE}/spectral.pth" \
    "${CKPT_BASE}/multiscale_pyramid.pth" \
    "${CKPT_BASE}/dual_branch.pth"; do
  if [[ -e "${f}" ]]; then
    printf "  ✓ %-60s (%s)\n" "${f#${REPO}/}" "$(du -h "${f}" 2>/dev/null | cut -f1)"
  else
    printf "  ✗ %-60s (missing)\n" "${f#${REPO}/}"
  fi
done

if [[ "${overall_status}" -eq 0 ]]; then
  echo "[train] ALL STEPS PASSED OR CACHED"
else
  echo "[train] ONE OR MORE STEPS FAILED — inspect ${LOG_DIR}/"
fi
exit "${overall_status}"
