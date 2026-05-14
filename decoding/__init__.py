"""Decoding package.

Subpackages:
  data/      dataset wrapper and split generation
  models/    all decoder architectures + the get_model() factory
  training/  training entry points (and training/jobs/ shell runners)
  common/    shared helpers (metrics, smoke-test fixtures)
  cli/       the `python -m decoding.cli` command-line interface

Scripts import the decoder as a package, e.g. `from decoding.models import
get_model`, after putting the repository root on sys.path. Evaluation,
analysis, and figure-generation code lives outside this package under the
top-level evaluation/ directory.
"""
