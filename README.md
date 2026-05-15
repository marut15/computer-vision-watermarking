# Computer-Vision Watermarking

Invisible watermarking of diffusion-generated images. An 8-bit identifier is
embedded into SDXL images via LoRA style-sliders (the **encoder**), and a
family of CNN / ViT / spectral **decoders** is trained to recover those bits
from the image alone. A separate **evaluation** stage measures clean accuracy,
robustness to image manipulations, and runs the diagnostic experiments and
figures used in the report.

## Repository structure

```
computer-vision-watermarking/
├── encoding/        Dataset generation: LoRA sliders + prompts + SDXL pipeline
├── decoding/        The decoder: data processing, model definitions, training
│   ├── data/        WatermarkDataset + train/val/test split generation
│   ├── models/      All decoder architectures + the get_model() factory
│   ├── training/    Training entry points (training/jobs/ = shell runners)
│   ├── common/      Shared helpers (metrics, smoke-test fixtures)
│   ├── configs/     Per-experiment YAML configs
│   ├── cli/         `python -m decoding.cli` command-line interface
│   └── tests/       Smoke / layout / pipeline tests
├── evaluation/      All evaluation, analysis and figure generation
│   ├── scripts/     evaluate.py, robustness_eval.py, compare_architectures.py
│   │   ├── analysis/  Diagnostic experiments (ablations, spectral probes, …)
│   │   ├── figures/   Paper-figure generators
│   │   └── jobs/      Shell runners that chain the above into full runs
│   ├── reports/     Written result write-ups (Markdown)
│   └── results/     Generated artifacts: figures/, metrics/, training_logs/
├── setup/           Workspace / S3 / environment bootstrap scripts
├── project_paths.py Resolves the data root (honours $PROJECT_DATA_ROOT)
└── requirements.txt
```

The decoder is importable as a package (`from decoding.models import
get_model`); evaluation code depends on `decoding`, never the other way round.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Image data and model checkpoints are **not** in git (too large), but they can be found at this link: https://drive.google.com/drive/folders/1Tg3HZLtD4kyKaET9nVvoxaKcYmKJCyus?usp=drive_link . Point the
pipeline at a local data root:

```bash
export PROJECT_DATA_ROOT=/path/to/data/computer-vision-watermarking
```

`project_paths.py` resolves all dataset / checkpoint paths from this variable;
configs reference it as `${PROJECT_DATA_ROOT}`. See `data_config.example.yaml`.

## Reproducing the experiments

```bash
# 0. (optional) end-to-end smoke test on a synthetic 64-image fixture, no GPU
bash decoding/tests/smoke_test.sh

# 1. generate the train/val/test split
python decoding/data/splits.py

# 2. train a decoder (any config under decoding/configs/)
python decoding/training/train.py --config decoding/configs/baseline_resnet50.yaml
#    or train all four global-pattern decoders:
bash decoding/training/jobs/train_new_decoders.sh

# 3. evaluate on the held-out test set
python evaluation/scripts/evaluate.py --config decoding/configs/baseline_resnet50.yaml

# 4. full evaluation suite (robustness + comparison + figures)
bash evaluation/scripts/jobs/run_full_evaluation.sh
```

Outputs land in `evaluation/results/` (figures, metrics, training logs) and
`evaluation/reports/` (Markdown write-ups).

## Quick checks

```bash
bash dry_run.sh        # syntax, structure, imports, configs — no GPU needed
```
