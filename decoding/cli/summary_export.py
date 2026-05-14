"""Collect existing JSON/MD/log summary artifacts into one folder."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Files we *expect* to find in a fully-populated repo. Anything missing is reported, never silently dropped.
EXPECTED_RESULTS: tuple[str, ...] = (
    "dual_branch_ablation.json",
    "robustness.json",
    "robustness_resnet.json",
    "robustness_separate.json",
    "robustness_vit.json",
    "architecture_comparison.md",
    "baseline_resnet50.md",
    "dual_branch_r50.md",
    "ablation1_efficientnet_b0.md",
    "ablation2_resolution_512.md",
    "decoder_performance.md",
)

EXPECTED_TRAINING_LOGS: tuple[str, ...] = (
    "dual_branch_r50.history.json",
    "dual_branch.eval.log",
    "dual_branch.robust.log",
    "global_stats.eval.log",
    "global_stats.robust.log",
    "spectral.eval.log",
    "spectral.robust.log",
    "multiscale_pyramid.eval.log",
    "multiscale_pyramid.robust.log",
)

EXPECTED_TEST_RESULTS: tuple[str, ...] = (
    "baseline_resnet50.json",
    "ablation1_efficientnet_b0.json",
    "ablation2_resolution_512.json",
)

EXPECTED_SCRIPTS: tuple[str, ...] = (
    "delta_stats.json",
)


@dataclass
class CollectedArtifact:
    src: Path
    dest_rel: str  # path under <output>/raw/
    kind: str      # "json" | "md" | "log" | "other"
    size_bytes: int


@dataclass
class ExportResult:
    output_dir: Path
    collected: List[CollectedArtifact] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    zip_path: Optional[Path] = None


def _kind(p: Path) -> str:
    s = p.suffix.lower()
    if s == ".json":
        return "json"
    if s in (".md", ".markdown"):
        return "md"
    if s == ".log":
        return "log"
    return "other"


def _copy_into(src: Path, dest_root: Path, rel: str) -> CollectedArtifact:
    dest = dest_root / "raw" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return CollectedArtifact(
        src=src,
        dest_rel=str(Path("raw") / rel),
        kind=_kind(src),
        size_bytes=src.stat().st_size,
    )


def collect_summaries(
    results_root: Path,
    output_dir: Path,
    scripts_dir: Optional[Path] = None,
    extra_paths: Optional[List[Path]] = None,
    make_zip: bool = False,
) -> ExportResult:
    """Copy all known summary artifacts under output_dir/raw/ and write an index."""
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    res = ExportResult(output_dir=output_dir)

    def _try(src: Path, rel: str) -> None:
        if src.is_file():
            res.collected.append(_copy_into(src, output_dir, rel))
        else:
            res.missing.append(str(src))

    for name in EXPECTED_RESULTS:
        _try(results_root / name, f"results/{name}")
    for name in EXPECTED_TRAINING_LOGS:
        _try(results_root / "training_logs" / name, f"results/training_logs/{name}")
    for name in EXPECTED_TEST_RESULTS:
        _try(results_root / "test_results" / name, f"results/test_results/{name}")
    if scripts_dir is not None:
        for name in EXPECTED_SCRIPTS:
            _try(Path(scripts_dir) / name, f"scripts/{name}")
    if extra_paths:
        for p in extra_paths:
            p = Path(p)
            if p.is_file():
                res.collected.append(_copy_into(p, output_dir, f"extra/{p.name}"))
            else:
                res.missing.append(str(p))

    write_index(res)
    write_report(res)

    if make_zip:
        zip_base = output_dir.with_suffix("")
        archive = shutil.make_archive(
            base_name=str(zip_base), format="zip", root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )
        res.zip_path = Path(archive)

    return res


def write_index(res: ExportResult) -> Path:
    payload: Dict[str, object] = {
        "output_dir": str(res.output_dir),
        "collected": [
            {
                "source": str(a.src),
                "dest": str(res.output_dir / a.dest_rel),
                "dest_rel": a.dest_rel,
                "kind": a.kind,
                "size_bytes": a.size_bytes,
            }
            for a in res.collected
        ],
        "missing": res.missing,
        "n_collected": len(res.collected),
        "n_missing": len(res.missing),
    }
    out = res.output_dir / "summary_index.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def write_report(res: ExportResult) -> Path:
    lines: List[str] = []
    lines.append("# Decoder summary export")
    lines.append("")
    lines.append(f"- Output directory: `{res.output_dir}`")
    lines.append(f"- Collected files: **{len(res.collected)}**")
    lines.append(f"- Missing files:   **{len(res.missing)}**")
    lines.append("")
    lines.append("## Collected")
    if res.collected:
        lines.append("")
        lines.append("| Kind | Size (B) | Path |")
        lines.append("| ---- | -------- | ---- |")
        for a in sorted(res.collected, key=lambda x: x.dest_rel):
            lines.append(f"| {a.kind} | {a.size_bytes} | `{a.dest_rel}` |")
    else:
        lines.append("")
        lines.append("_(none)_")
    lines.append("")
    lines.append("## Missing")
    if res.missing:
        lines.append("")
        for m in sorted(res.missing):
            lines.append(f"- `{m}`")
    else:
        lines.append("")
        lines.append("_(none — every expected file was found)_")
    out = res.output_dir / "summary_report.md"
    out.write_text("\n".join(lines) + "\n")
    return out
