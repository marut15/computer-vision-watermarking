"""`decoding.cli s3-plan` — propose the canonical S3 layout (no uploads)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .. import DECODING_ROOT, CHECKPOINTS_ROOT, MODEL_BUNDLES_ROOT
from ..checkpoints import (
    file_size_bytes,
    resolve_checkpoint,
    sha256_of,
)
from ..registry import MODELS, ModelSpec
from ..s3 import join_s3


DEFAULT_BUCKET = "s3://<bucket>/computer-vision-watermarking"


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bucket", default=DEFAULT_BUCKET,
                   help="Root S3 URI used in the proposal "
                        f"(default: {DEFAULT_BUCKET})")
    p.add_argument("--weights-root", default=str(MODEL_BUNDLES_ROOT),
                   help="Local root used to compute sha256/size for inventory. "
                        f"Default: $PROJECT_DATA_ROOT/decoding/model_bundles")
    p.add_argument("--results-root", default=str(DECODING_ROOT / "results"))
    p.add_argument("--output", required=True,
                   help="Directory to write s3_structure_proposal.{md,json}.")
    p.add_argument("--no-hash", action="store_true",
                   help="Skip sha256 computation (faster for huge files).")


def _proposed_tree(bucket: str) -> Dict[str, object]:
    """The canonical S3 layout, expressed as a nested dict tree."""
    return {
        bucket.rstrip("/") + "/": {
            "decoding/": {
                "checkpoints/": {
                    "baseline/": [
                        "baseline_resnet50.pth",
                        "config.yaml",
                        "metrics.json",
                    ],
                    "dual_branch_r50/": [
                        "dual_branch_r50.pth",
                        "dual_branch_r50.yaml",
                        "training.log",
                        "training_history.json",
                        "clean_metrics.json",
                        "ablation.json",
                        "robustness.json",
                        "figures/",
                    ],
                    "dual_branch/": ["dual_branch.pth", "dual_branch.yaml",
                                     "training.log", "clean_metrics.json",
                                     "ablation.json", "robustness.json"],
                    "spectral/": ["spectral.pth", "spectral.yaml",
                                  "training.log", "clean_metrics.json",
                                  "robustness.json"],
                    "global_stats/": ["global_stats.pth", "global_stats.yaml",
                                      "training.log", "clean_metrics.json",
                                      "robustness.json"],
                    "multiscale_pyramid/": ["multiscale_pyramid.pth",
                                            "multiscale.yaml",
                                            "training.log",
                                            "robustness.json"],
                    "efficientnet_b0/": ["efficientnet_b0.pth",
                                         "ablation1_efficientnet_b0.yaml",
                                         "clean_metrics.json"],
                },
                "results/": {
                    "summaries/": ["summary_index.json", "summary_report.md"],
                    "comparison_figures/": [
                        "architecture_comparison.png",
                        "architecture_per_bit.png",
                        "overall_comparison.png",
                    ],
                    "robustness/": [
                        "robustness.json",
                        "robustness_resnet.json",
                        "robustness_separate.json",
                        "robustness_vit.json",
                    ],
                },
                "datasets/": [
                    "splits.json",
                    "metadata.json",
                    "images_manifest.json",
                ],
            },
            "encoding/": {
                "checkpoints/": ["lora_sliders/"],
                "outputs/": ["sample_grids/"],
            },
            "manifests/": [
                "decoder_manifest.json",
                "s3_inventory.json",
            ],
        }
    }


def _render_tree(tree: Dict[str, object], indent: int = 0) -> List[str]:
    lines: List[str] = []
    pad = "  " * indent
    for key, val in tree.items():
        lines.append(f"{pad}{key}")
        if isinstance(val, dict):
            lines.extend(_render_tree(val, indent + 1))
        elif isinstance(val, list):
            for v in val:
                lines.append(f"{pad}  {v}")
    return lines


def _checkpoint_manifest(
    spec: ModelSpec,
    weights_root: Path,
    bucket: str,
    results_root: Path,
    include_hash: bool,
) -> Dict[str, object]:
    ck = resolve_checkpoint(spec, weights_root)
    entry: Dict[str, object] = {
        "model_name": spec.name,
        "architecture": spec.name,
        "description": spec.description,
        "image_size": spec.image_size,
        "expected_num_bits": 8,
        "supports_branch_weights": spec.supports_branch_weights,
        "checkpoint_s3_uri": join_s3(bucket, "decoding/checkpoints",
                                     spec.name, spec.checkpoint_filename),
        "config_s3_uri": (
            join_s3(bucket, "decoding/checkpoints", spec.name, spec.config)
            if spec.config else None
        ),
        "local_checkpoint": str(ck.path) if ck.path else None,
        "local_checkpoint_found": ck.path is not None,
        "tried_paths": [str(p) for p in ck.candidates_tried],
        "training_date": None,
        "clean_metrics": None,
        "robustness_metrics": None,
        "sha256": None,
        "file_size_bytes": None,
        "notes": None,
    }
    # Best-effort local enrichment for the inventory.
    if ck.path and ck.path.is_file():
        entry["file_size_bytes"] = file_size_bytes(ck.path)
        if include_hash:
            entry["sha256"] = sha256_of(ck.path)
        try:
            entry["training_date"] = time.strftime(
                "%Y-%m-%d", time.gmtime(ck.path.stat().st_mtime)
            )
        except OSError:
            pass

    clean_md = results_root / f"{spec.name}.md"
    if clean_md.is_file():
        entry["notes"] = f"see {clean_md.name}"

    test_json = results_root / "test_results" / f"{spec.name}.json"
    if test_json.is_file():
        try:
            entry["clean_metrics"] = json.loads(test_json.read_text()).get("test_metrics")
        except json.JSONDecodeError:
            pass
    return entry


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    tree = _proposed_tree(args.bucket)
    rendered = "\n".join(_render_tree(tree))

    weights_root = Path(args.weights_root).expanduser()
    results_root = Path(args.results_root).expanduser()
    inventory = [
        _checkpoint_manifest(spec, weights_root, args.bucket, results_root,
                             include_hash=not args.no_hash)
        for spec in MODELS
    ]

    proposal_json = {
        "bucket": args.bucket,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tree": tree,
        "decoder_manifest": inventory,
    }
    (output / "s3_structure_proposal.json").write_text(json.dumps(proposal_json, indent=2))

    md_lines: List[str] = []
    md_lines.append("# Proposed S3 layout")
    md_lines.append("")
    md_lines.append(f"Bucket root: `{args.bucket}`")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(rendered)
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Decoder manifest")
    md_lines.append("")
    md_lines.append("| model | image_size | branch_weights | local_found | s3_uri |")
    md_lines.append("| ----- | ---------- | -------------- | ----------- | ------ |")
    for e in inventory:
        md_lines.append(
            f"| {e['model_name']} | {e['image_size']} | "
            f"{'yes' if e['supports_branch_weights'] else 'no'} | "
            f"{'yes' if e['local_checkpoint_found'] else 'no'} | "
            f"`{e['checkpoint_s3_uri']}` |"
        )
    (output / "s3_structure_proposal.md").write_text("\n".join(md_lines) + "\n")

    print(f"wrote {output / 's3_structure_proposal.json'}")
    print(f"wrote {output / 's3_structure_proposal.md'}")
    return 0
