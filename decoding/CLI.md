# Decoder CLI

A single entry point — `python -m decoding.cli` — for evaluating decoder
models, running decoders on watermarked images, exporting existing
summaries, and proposing the final S3 layout. The CLI is a thin wrapper
around the existing `src/dataloader.py`, `src/models/`, `src/utils.py`,
and `scripts/` machinery; it does not duplicate any model, loader, or
metric code.

```
python -m decoding.cli --help
```

```
COMMAND
  test              evaluate decoder model(s) on a split
  decode            decode watermarked images
  export-summaries  bundle existing summary artifacts
  s3-plan           emit the proposed S3 layout (JSON + Markdown)
  clean             remove obviously stale local files (safe by default)
```

## Model names and aliases

Resolved by [`decoding/cli/registry.py`](cli/registry.py). The factory
in [`decoding/src/models/__init__.py`](src/models/__init__.py) is the source
of truth; the registry only adds CLI-friendly aliases.

| Canonical | Aliases | Default image size | Default config |
| --------- | ------- | ------------------ | -------------- |
| `resnet50` | `baseline`, `baseline_resnet50` | 1024 | `baseline_resnet50.yaml` |
| `efficientnet_b0` | `efficientnet`, `effnet_b0` | 1024 | `ablation1_efficientnet_b0.yaml` |
| `global_stats` | `stats`, `globalstats` | 256 | `global_stats.yaml` |
| `spectral` | `fft`, `spectrum` | 1024 | `spectral.yaml` |
| `multiscale_pyramid` | `multiscale`, `pyramid` | 512 | `multiscale.yaml` |
| `dual_branch` | `dualbranch` | 512 | `dual_branch.yaml` |
| `dual_branch_r34` | `dual_branch_resnet34` | 512 | (falls back to dual_branch.yaml) |
| `dual_branch_r50` | `dual_branch_resnet50` | 1024 | `dual_branch_r50.yaml` |

`--models all` runs every registered model in the order above.
`--models baseline,dual_branch_r50,spectral` picks specific ones (deduped).

## Checkpoint discovery

`--weights-root <root>` is searched in this order for each model `<arch>`:

1. `<root>/<arch>/<arch>.pth` — the S3 staging layout
   (e.g. `/workspace/new_models/dual_branch_r50/dual_branch_r50.pth`).
2. `<root>/<arch>.pth` — flat layout (e.g. `decoding/checkpoints/global_stats.pth`).
3. `<root>/<legacy_filename>` — for `resnet50` this is `baseline_resnet50.pth`.
4. `<root>/<arch>/model.pth`.

Per-model overrides win over the search: `--checkpoint dual_branch_r50=/path/to.pth`.
Repeatable.

## Subcommand: `test`

```
python -m decoding.cli test \
  --models all \
  --weights-root /workspace/new_models \
  --metadata watermark_encoding/data/metadata.json \
  --images watermark_encoding/data/images \
  --splits decoding/data/splits.json \
  --batch-size 8 \
  --device auto \
  --output decoding/results/cli_eval
```

Writes:
- `<output>/<arch>.json` — per-model metrics.
- `<output>/summary.json` and `<output>/summary.md`.

Useful flags:
- `--dry-run` — print the plan (which model + which checkpoint + which config)
  and exit. Required when you want to confirm checkpoint resolution before
  burning GPU time.
- `--limit N` — cap the number of evaluated images (debugging).
- `--image-size N` — override per-model defaults.
- `--threshold 0.5` — change the binarization threshold for predictions.

### Dual-branch weighting

For `dual_branch`, `dual_branch_r34`, `dual_branch_r50` only. Applied via
the new `forward_with_branch_weights(...)` method on `DualBranchDecoder`
which scales pooled feature vectors before fusion. **Identical to the
existing `forward` when both weights are 1.0.**

```
# Spectral-only equivalent (zero out spatial branch)
python -m decoding.cli test \
  --models dual_branch_r50 \
  --dual-spatial-weight 0.0 \
  --dual-spectral-weight 1.0 \
  --weights-root /workspace/new_models \
  --output decoding/results/dual_branch_spectral_only

# Compact syntax
python -m decoding.cli test --models dual_branch_r50 \
  --dual-weights spatial=0.25,spectral=1.0 \
  --weights-root /workspace/new_models \
  --output decoding/results/dual_branch_attenuated_spatial
```

Mathematically equivalent to the modes in
[`scripts/ablate_dual_branch.py`](scripts/ablate_dual_branch.py):
- `spatial=0, spectral=1` ⇔ `no_spatial`
- `spatial=1, spectral=0` ⇔ `no_spectral`
- `spatial=1, spectral=1` ⇔ `full` (default)

## Subcommand: `decode`

```
python -m decoding.cli decode \
  --model dual_branch_r50 \
  --input watermark_encoding/data/images \
  --weights-root /workspace/new_models \
  --output decoding/outputs/decoded_dual_branch_r50 \
  --device auto \
  --format json,csv,md
```

- `--input` accepts a single image or a directory (recursive, png/jpg/jpeg/webp).
- `--format` is a comma-separated subset of `json`, `csv`, `md`. Default `json`.
- Missing/corrupt images are recorded with a `status` field; the CLI never
  bails on a single bad file.
- Each record contains: filename, abs path, model, checkpoint, predicted
  bits, probabilities, logits, threshold, branch weights (if applicable),
  and a timestamp.

### Pointing at an explicit checkpoint

```
python -m decoding.cli decode \
  --model baseline \
  --input some_images/ \
  --checkpoint decoding/checkpoints/baseline_resnet50.pth \
  --output decoding/outputs/baseline_decode
```

### Optional S3 download

```
python -m decoding.cli decode \
  --model dual_branch_r50 \
  --input some_images/ \
  --weights-root /workspace/new_models \
  --s3-uri s3://my-bucket/computer-vision-watermarking/decoding/checkpoints/dual_branch_r50/dual_branch_r50.pth \
  --download-missing \
  --output decoding/outputs/decoded
```

The S3 path is only touched when the local checkpoint cannot be resolved
**and** `--download-missing` is set. `boto3` is imported lazily — installing
it is only required for this code path.

## Subcommand: `export-summaries`

Bundles every committed summary artifact into one folder. Missing files are
reported, never silently dropped.

```
python -m decoding.cli export-summaries \
  --results-root decoding/results \
  --output decoding/exports/latest \
  --zip
```

Output:
- `<output>/raw/...` — copies of every JSON / MD / log under
  `decoding/results/`, `decoding/results/training_logs/`,
  `decoding/results/test_results/`, plus `decoding/scripts/delta_stats.json`.
- `<output>/summary_index.json` — collected vs. missing.
- `<output>/summary_report.md` — human-readable index.
- `<output>.zip` — optional, when `--zip` is passed.

`--extra PATH` adds arbitrary extra files (repeatable).

## Subcommand: `s3-plan`

Generates the proposed S3 layout and a per-model manifest. **No upload.**

```
python -m decoding.cli s3-plan \
  --weights-root /workspace/new_models \
  --results-root decoding/results \
  --bucket s3://my-bucket/computer-vision-watermarking \
  --output decoding/exports/s3_plan
```

Writes `s3_structure_proposal.md` and `s3_structure_proposal.json`. The JSON
contains a `decoder_manifest` array with one entry per registered model:

```json
{
  "model_name": "dual_branch_r50",
  "architecture": "dual_branch_r50",
  "image_size": 1024,
  "supports_branch_weights": true,
  "checkpoint_s3_uri": "s3://.../decoding/checkpoints/dual_branch_r50/dual_branch_r50.pth",
  "config_s3_uri":     "s3://.../decoding/checkpoints/dual_branch_r50/dual_branch_r50.yaml",
  "local_checkpoint": null,
  "local_checkpoint_found": false,
  "tried_paths": ["/workspace/new_models/dual_branch_r50/dual_branch_r50.pth", "..."],
  "training_date": null,
  "clean_metrics": null,
  "robustness_metrics": null,
  "sha256": null,
  "file_size_bytes": null,
  "notes": null
}
```

When checkpoints exist locally under `--weights-root`, the entry is
enriched with `sha256` (skip with `--no-hash` if computing is too slow),
`file_size_bytes`, and `training_date` derived from mtime.

## Subcommand: `clean`

Safe local cleanup. Default is dry-run — pass `--yes` to actually delete.

```
# inspect
python -m decoding.cli clean --dry-run

# actually delete
python -m decoding.cli clean --yes
```

Always-safe targets:
- `__pycache__/` directories
- `.ipynb_checkpoints/` directories
- `.DS_Store`, `Thumbs.db`
- `*.pyc`, `*.pyo`

Protected by default — pass an explicit flag to include:
- `.pth` / `.pt` / `.safetensors`: `--delete-weights`
- Generated figures under `decoding/figures/` or `decoding/results/figures/`:
  `--delete-figures`

Always skipped:
- `.git/`, `decoding/configs/`, `decoding/data/` (committed configs and splits)
- Anything outside `--root` (defaults to repo root).

## Tests

```
python -m unittest decoding.tests.test_cli_smoke
```

22 stdlib-only smoke tests covering help, alias resolution, checkpoint
discovery, `s3-plan`, `export-summaries`, and `clean`. No torch, no PIL,
no S3 credentials needed.

For the heavy paths (`test` and `decode` actually running models) you need
torch + torchvision + PIL + numpy + the dataset. The recommended way is
the existing `decoding/scripts/run_full_evaluation.sh` flow on Runpod, then
verify the CLI behaves the same way:

```
python -m decoding.cli test --models all --dry-run \
  --weights-root /workspace/new_models \
  --output /tmp/cli_dry
```

Confirm every entry says `ckpt=OK` before kicking off the real eval.
