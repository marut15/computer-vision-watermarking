#!/usr/bin/env bash
# Dry-run / self-test for the repository layout.
#
# Verifies — without a GPU and without the dataset — that the reorganised
# repo is internally consistent: every script compiles, every shell script
# parses, the new directory structure is in place, no code still references
# the old paths, all configs load, and every `from decoding.*` import
# resolves to a real module. When PyTorch is installed it additionally
# imports the decoder package, builds every model, and exercises the CLI.
#
# Usage:  bash dry_run.sh
# Exit 0 = all checks passed, 1 = one or more failed.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$(pwd)"
PYTHON="${PYTHON:-python3}"

GREEN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; RST=$'\033[0m'
fail_count=0
pass()  { echo "  ${GREEN}PASS${RST}  $1"; }
fail()  { echo "  ${RED}FAIL${RST}  $1"; [[ -n "${2:-}" ]] && echo "        $2"; fail_count=$((fail_count + 1)); }
skip()  { echo "  ${YEL}SKIP${RST}  $1"; }
section() { echo; echo "── $1 ──"; }

# ── 1. Python compiles ────────────────────────────────────────────────────────
section "1. Python syntax (py_compile)"
if out="$(find decoding evaluation encoding -name '*.py' -print0 | xargs -0 "${PYTHON}" -m py_compile 2>&1)"; then
  pass "all Python files compile"
else
  fail "py_compile errors" "${out}"
fi

# ── 2. Shell scripts parse ────────────────────────────────────────────────────
section "2. Shell syntax (bash -n)"
sh_bad=0
while IFS= read -r f; do
  bash -n "$f" 2>/dev/null || { fail "bash -n $f"; sh_bad=1; }
done < <(find decoding evaluation setup -name '*.sh')
[[ ${sh_bad} -eq 0 ]] && pass "all shell scripts parse"

# ── 3. Directory structure ────────────────────────────────────────────────────
section "3. Directory structure"
expect_exist=(
  decoding/__init__.py decoding/data/__init__.py decoding/data/dataset.py
  decoding/data/splits.py decoding/data/splits.json
  decoding/models/__init__.py decoding/models/separate.py decoding/models/vit.py
  decoding/models/dual_branch.py decoding/common/__init__.py
  decoding/common/metrics.py decoding/common/smoke.py
  decoding/training/train.py decoding/training/jobs/train_new_decoders.sh
  decoding/tests/test_pipeline.py decoding/tests/smoke_test.sh
  evaluation/scripts/evaluate.py evaluation/scripts/robustness_eval.py
  evaluation/scripts/compare_architectures.py
  evaluation/scripts/analysis/ablate_dual_branch.py
  evaluation/scripts/figures/create_figures.py
  evaluation/scripts/jobs/run_full_evaluation.sh
  evaluation/reports evaluation/results/figures evaluation/results/metrics
  README.md decoding/README.md evaluation/README.md
)
expect_gone=( decoding/src decoding/scripts decoding/analysis decoding/figures
              decoding/results decoding/_smoke_utils.py setup_environment )
struct_ok=1
for p in "${expect_exist[@]}"; do
  [[ -e "$p" ]] || { fail "missing: $p"; struct_ok=0; }
done
for p in "${expect_gone[@]}"; do
  [[ -e "$p" ]] && { fail "should not exist: $p"; struct_ok=0; }
done
[[ ${struct_ok} -eq 1 ]] && pass "new layout present, old layout removed"

# ── 4. No stale references ────────────────────────────────────────────────────
section "4. Stale path / import references"
stale="$(grep -rn -E "from src\.|import src$|from _smoke_utils|decoding/src/|decoding/scripts/|decoding/analysis/" \
          --include='*.py' --include='*.sh' decoding evaluation setup 2>/dev/null || true)"
if [[ -z "${stale}" ]]; then
  pass "no references to pre-reorg paths"
else
  fail "stale references found" "${stale}"
fi

# ── 5. Configs load + path tokens ─────────────────────────────────────────────
section "5. Decoder configs"
cfg_out="$("${PYTHON}" - <<'PY'
import glob, sys
try:
    import yaml
except ImportError:
    print("SKIP pyyaml not installed"); sys.exit(0)
bad = []
for f in sorted(glob.glob("decoding/configs/*.yaml")):
    try:
        c = yaml.safe_load(open(f))
        assert c["data"]["splits_path"], f"{f}: empty splits_path"
        assert c["output"]["results"].startswith("evaluation/"), f"{f}: output.results not under evaluation/"
        assert c["output"]["checkpoint"], f"{f}: empty checkpoint"
    except Exception as e:
        bad.append(str(e))
if bad:
    print("FAIL " + " | ".join(bad)); sys.exit(1)
print(f"OK {len(glob.glob('decoding/configs/*.yaml'))} configs parse and point under evaluation/")
PY
)"
case "${cfg_out}" in
  OK*)   pass "${cfg_out#OK }" ;;
  SKIP*) skip "${cfg_out#SKIP }" ;;
  *)     fail "config check" "${cfg_out#FAIL }" ;;
esac

# ── 6. Static import-graph resolution ─────────────────────────────────────────
section "6. decoding.* imports resolve to real modules"
imp_out="$("${PYTHON}" - <<'PY'
import ast, pathlib, sys
repo = pathlib.Path(".")
bad = []
def resolves(mod):
    parts = mod.split(".")
    base = repo.joinpath(*parts)
    return (base.with_suffix(".py").exists()
            or (base / "__init__.py").exists())
for f in list(repo.glob("decoding/**/*.py")) + list(repo.glob("evaluation/**/*.py")):
    try:
        tree = ast.parse(f.read_text(), filename=str(f))
    except SyntaxError as e:
        bad.append(f"{f}: syntax {e}"); continue
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        for m in mods:
            if m == "decoding" or m.startswith("decoding."):
                if not resolves(m):
                    bad.append(f"{f}:{node.lineno}: unresolved import '{m}'")
if bad:
    print("FAIL\n  " + "\n  ".join(bad)); sys.exit(1)
print("OK every decoding.* import maps to an existing module")
PY
)"
case "${imp_out}" in
  OK*) pass "${imp_out#OK }" ;;
  *)   fail "import-graph check" "${imp_out#FAIL}" ;;
esac

# ── 7. Runtime smoke (needs PyTorch) ──────────────────────────────────────────
section "7. Runtime import / model build (needs PyTorch)"
if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
  skip "PyTorch not installed — skipping package import, model build, CLI, smoke fixture"
else
  if rt="$("${PYTHON}" - <<'PY' 2>&1
import decoding
from decoding.data.dataset import WatermarkDataset
from decoding.common.metrics import compute_metrics, print_metrics
from decoding.common.smoke import ensure_smoke_fixture, pick_device
from decoding.models import get_model
for a in ["resnet50","efficientnet_b0","global_stats","spectral",
          "multiscale_pyramid","dual_branch","dual_branch_r50"]:
    get_model(a, pretrained=False)
from decoding.models.separate import SeparateBitClassifier
from decoding.models.vit import ViTWatermarkDecoder
fx = ensure_smoke_fixture(root="decoding/.smoke")
assert fx.metadata.exists() and fx.splits.exists()
print("OK")
PY
)"; [[ "${rt}" == *OK ]]; then
    pass "decoding package imports, all 7 architectures build, smoke fixture OK"
  else
    fail "runtime import / model build" "${rt}"
  fi
  if "${PYTHON}" -m decoding.cli --help >/dev/null 2>&1; then
    pass "python -m decoding.cli --help"
  else
    fail "python -m decoding.cli --help"
  fi
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo
if [[ ${fail_count} -eq 0 ]]; then
  echo "${GREEN}═══ dry-run: ALL CHECKS PASSED ═══${RST}"
  exit 0
else
  echo "${RED}═══ dry-run: ${fail_count} CHECK(S) FAILED ═══${RST}"
  exit 1
fi
