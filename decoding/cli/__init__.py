"""Decoder CLI entry point.

Subcommands:
  test              evaluate one or more decoder models on a test split.
  decode            run a trained decoder on watermarked images.
  export-summaries  collect existing JSON/MD/log artifacts into one folder.
  s3-plan           emit the proposed S3 layout (JSON + Markdown).
  clean             remove obviously stale local artifacts (safe by default).

This package adds `decoding/` to sys.path on import so the existing
`from src.models import get_model` pattern used by every script in
decoding/scripts/ continues to work unchanged from inside the CLI modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

DECODING_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = DECODING_ROOT.parent

# Match the convention used in decoding/scripts/*.py so we can reuse the
# existing dataloader, model factory, and metric helpers without rewriting
# anything. Idempotent.
_decoding_str = str(DECODING_ROOT)
if _decoding_str not in sys.path:
    sys.path.insert(0, _decoding_str)

# Resolve the data root via project_paths so PROJECT_DATA_ROOT env var works.
sys.path.insert(0, str(REPO_ROOT))
from project_paths import Paths as _Paths  # noqa: E402

_paths = _Paths()
DATA_ROOT = _paths.data_root          # /workspace/data/computer-vision-watermarking
CHECKPOINTS_ROOT = _paths.decoder_checkpoints   # …/decoding/checkpoints
MODEL_BUNDLES_ROOT = _paths.model_bundles        # …/decoding/model_bundles
