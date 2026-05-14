# evaluation/

Everything that measures the decoder or produces artifacts for the report:
test-set evaluation, robustness testing, diagnostic analysis experiments, and
figure generation. Code here imports the decoder as a package
(`from decoding.models import get_model`); the decoder never imports this.

## Layout

| Path | Contents |
|------|----------|
| `scripts/evaluate.py` | Clean test-set evaluation for one config |
| `scripts/robustness_eval.py` | JPEG / resize / crop attacks against a decoder |
| `scripts/compare_architectures.py` | Side-by-side test-set comparison + table/chart |
| `scripts/analysis/` | Diagnostic experiments — branch ablations, spectral/radial probes, Δ-statistics, synthetic controls |
| `scripts/figures/` | Paper-figure generators |
| `scripts/jobs/` | Shell runners chaining the above into full resumable runs |
| `reports/` | Written result write-ups (Markdown) |
| `results/figures/` | Generated plots (`.png`) |
| `results/metrics/` | Generated metrics (`.json`, incl. `test_results/`) |
| `results/training_logs/` | Per-epoch training / eval logs |

## Usage

```bash
# single-config test-set evaluation
python evaluation/scripts/evaluate.py --config decoding/configs/dual_branch.yaml

# robustness suite for one model
python evaluation/scripts/robustness_eval.py --model resnet

# full evaluation pipeline (robustness + comparison + signal figures)
bash evaluation/scripts/jobs/run_full_evaluation.sh

# evaluate the four new decoders and stage results
bash evaluation/scripts/jobs/eval_new_decoders.sh

# one diagnostic experiment
python evaluation/scripts/analysis/ablate_dual_branch.py \
  --config decoding/configs/dual_branch.yaml
```

All scripts resolve their own paths from `__file__`, so they can be run from
any working directory. Generated outputs always land under `results/`.
