"""Canonical decoder registry: names, aliases, configs, default weights.

Single source of truth used by every CLI subcommand. Mirrors
decoding/src/models/__init__.py — do not invent architectures here that the
factory does not support; the test command resolves names through this table
and then calls `get_model(canonical_name, ...)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    name: str                       # canonical name passed to get_model()
    aliases: tuple[str, ...]        # CLI-friendly aliases (case-insensitive)
    config: Optional[str]           # filename under decoding/configs/
    checkpoint_filename: str        # default `<name>.pth` file under weights-root/<name>/
    legacy_checkpoint: Optional[str] = None  # fallback under decoding/checkpoints/
    image_size: int = 1024
    supports_branch_weights: bool = False
    description: str = ""


# Order matters for `--models all`: keep "cheapest first" so dry-runs and
# smoke tests touch the lightweight models before the R-50 trunks.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="global_stats",
        aliases=("global_stats", "globalstats", "stats"),
        config="global_stats.yaml",
        checkpoint_filename="global_stats.pth",
        legacy_checkpoint="global_stats.pth",
        image_size=256,
        description="MLP over hand-crafted global image statistics.",
    ),
    ModelSpec(
        name="spectral",
        aliases=("spectral", "fft", "spectrum"),
        config="spectral.yaml",
        checkpoint_filename="spectral.pth",
        legacy_checkpoint="spectral.pth",
        image_size=1024,
        description="CNN over the 2D-FFT log-magnitude.",
    ),
    ModelSpec(
        name="efficientnet_b0",
        aliases=("efficientnet_b0", "efficientnet", "effnet_b0", "ablation1_efficientnet_b0"),
        config="ablation1_efficientnet_b0.yaml",
        checkpoint_filename="efficientnet_b0.pth",
        legacy_checkpoint="ablation1_efficientnet_b0.pth",
        image_size=1024,
        description="EfficientNet-B0 classifier head.",
    ),
    ModelSpec(
        name="resnet50",
        aliases=("resnet50", "baseline", "baseline_resnet50", "resnet"),
        config="baseline_resnet50.yaml",
        checkpoint_filename="baseline_resnet50.pth",
        legacy_checkpoint="baseline_resnet50.pth",
        image_size=1024,
        description="Person A's shared-backbone ResNet-50 baseline.",
    ),
    ModelSpec(
        name="multiscale_pyramid",
        aliases=("multiscale_pyramid", "multiscale", "pyramid"),
        config="multiscale.yaml",
        checkpoint_filename="multiscale_pyramid.pth",
        legacy_checkpoint="multiscale_pyramid.pth",
        image_size=512,
        description="Image-pyramid ResNet-18 with avg/std/max fusion.",
    ),
    ModelSpec(
        name="dual_branch",
        aliases=("dual_branch", "dualbranch"),
        config="dual_branch.yaml",
        checkpoint_filename="dual_branch.pth",
        legacy_checkpoint="dual_branch.pth",
        image_size=512,
        supports_branch_weights=True,
        description="ResNet-18 spatial + FFT spectral branches.",
    ),
    ModelSpec(
        name="dual_branch_r34",
        aliases=("dual_branch_r34", "dual_branch_resnet34"),
        config=None,  # not committed; falls back to dual_branch.yaml at runtime
        checkpoint_filename="dual_branch_r34.pth",
        legacy_checkpoint="dual_branch_r34.pth",
        image_size=512,
        supports_branch_weights=True,
        description="DualBranch with ResNet-34 spatial trunk.",
    ),
    ModelSpec(
        name="dual_branch_r50",
        aliases=("dual_branch_r50", "dual_branch_resnet50"),
        config="dual_branch_r50.yaml",
        checkpoint_filename="dual_branch_r50.pth",
        legacy_checkpoint="dual_branch_r50.pth",
        image_size=1024,
        supports_branch_weights=True,
        description="DualBranch with ResNet-50 spatial trunk.",
    ),
)


def _alias_index() -> Dict[str, ModelSpec]:
    idx: Dict[str, ModelSpec] = {}
    for spec in MODELS:
        for a in (spec.name, *spec.aliases):
            idx[a.lower()] = spec
    return idx


_INDEX = _alias_index()


def all_model_names() -> List[str]:
    return [m.name for m in MODELS]


def resolve(name: str) -> ModelSpec:
    """Map a user-supplied name/alias to its canonical ModelSpec.

    Raises ValueError with the known set on miss.
    """
    key = name.strip().lower()
    if key not in _INDEX:
        known = ", ".join(sorted({m.name for m in MODELS}))
        raise ValueError(f"unknown model {name!r}; known: {known}")
    return _INDEX[key]


def parse_models_arg(arg: str) -> List[ModelSpec]:
    """Parse `--models all` or `--models a,b,c` into a list of unique specs.

    Preserves the order the user supplied (or registry order for 'all').
    """
    if arg.strip().lower() in ("all", "*"):
        return list(MODELS)
    seen: set[str] = set()
    out: List[ModelSpec] = []
    for raw in arg.split(","):
        token = raw.strip()
        if not token:
            continue
        spec = resolve(token)
        if spec.name in seen:
            continue
        seen.add(spec.name)
        out.append(spec)
    if not out:
        raise ValueError("no models selected; pass --models all or a non-empty list")
    return out
