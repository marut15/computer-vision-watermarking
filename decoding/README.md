# decoding/

The watermark decoder: data processing, model definitions, and training. This
directory is a Python package — import it as `decoding.*` with the repository
root on `sys.path`. Evaluation, analysis, and figure generation live outside
it under the top-level `evaluation/` directory.

## Layout

| Path | Contents |
|------|----------|
| `data/` | `dataset.py` (`WatermarkDataset`) and `splits.py` (stratified train/val/test split → `data/splits.json`) |
| `models/` | All architectures — `resnet`, `efficientnet`, `separate`, `vit`, `global_stats`, `spectral`, `multiscale`, `dual_branch` — plus the `get_model()` factory in `__init__.py` |
| `training/` | `train.py` (generic config-driven trainer), `train_separate.py`, `train_vit.py`, `train_dual_branch_efficient.py`; `jobs/` holds shell runners |
| `common/` | `metrics.py` (per-bit accuracy, exact-match) and `smoke.py` (synthetic fixture for GPU-free smoke tests) |
| `configs/` | One YAML per experiment; data paths use `${PROJECT_DATA_ROOT}` |
| `cli/` | `python -m decoding.cli` — `test`, `decode`, `export-summaries`, `s3-plan`, `clean` |
| `tests/` | `smoke_test.sh`, `test_layout.py`, `test_pipeline.py` |
| `checkpoints/` | Trained weights (git-ignored; populated from S3 or training) |

## Usage

```bash
# generate the split
python decoding/data/splits.py

# train (run from anywhere — scripts self-locate the repo root)
python decoding/training/train.py --config decoding/configs/dual_branch.yaml

# train all four global-pattern decoders, resumable
bash decoding/training/jobs/train_new_decoders.sh

# CLI
python -m decoding.cli test --help
```

Config conventions: `data.splits_path` is resolved relative to `decoding/`;
`data.metadata_path` / `images_path` / `output.checkpoint` use the
`${PROJECT_DATA_ROOT}` token expanded at load time.
