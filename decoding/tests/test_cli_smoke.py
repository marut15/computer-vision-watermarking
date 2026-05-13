"""Decoder CLI smoke tests.

stdlib-only on purpose — runnable as:
    python -m unittest decoding.tests.test_cli_smoke
without pytest, torch, PIL, or numpy installed. Covers the plumbing that
does *not* depend on real model weights or the watermark dataset.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

# Ensure the repo root is on sys.path even when this is run from elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoding.cli import registry  # noqa: E402
from decoding.cli.main import main as cli_main  # noqa: E402
from decoding.cli.commands import clean as clean_cmd  # noqa: E402
from decoding.cli.commands import export_summaries  # noqa: E402
from decoding.cli.commands import s3_plan  # noqa: E402
from decoding.cli.commands import test_cmd  # noqa: E402
from decoding.cli.checkpoints import parse_overrides, resolve_checkpoint  # noqa: E402


PY = sys.executable


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke `python -m decoding.cli ...` from the repo root."""
    return subprocess.run(
        [PY, "-m", "decoding.cli", *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


class TestCliHelp(unittest.TestCase):
    def test_top_level_help(self):
        cp = _run_cli("--help")
        self.assertEqual(cp.returncode, 0, msg=cp.stderr)
        self.assertIn("test", cp.stdout)
        self.assertIn("decode", cp.stdout)
        self.assertIn("export-summaries", cp.stdout)
        self.assertIn("s3-plan", cp.stdout)
        self.assertIn("clean", cp.stdout)

    def test_subcommand_help(self):
        for sub in ("test", "decode", "export-summaries", "s3-plan", "clean"):
            with self.subTest(sub=sub):
                cp = _run_cli(sub, "--help")
                self.assertEqual(cp.returncode, 0, msg=cp.stderr)


class TestRegistry(unittest.TestCase):
    def test_canonical_names_present(self):
        names = registry.all_model_names()
        for required in ("resnet50", "efficientnet_b0", "global_stats", "spectral",
                         "multiscale_pyramid", "dual_branch",
                         "dual_branch_r34", "dual_branch_r50"):
            self.assertIn(required, names)

    def test_baseline_alias_maps_to_resnet50(self):
        self.assertEqual(registry.resolve("baseline").name, "resnet50")
        self.assertEqual(registry.resolve("BASELINE").name, "resnet50")

    def test_dual_branch_aliases(self):
        for alias in ("dual_branch_r50", "dual_branch_resnet50"):
            self.assertEqual(registry.resolve(alias).name, "dual_branch_r50")
            self.assertTrue(registry.resolve(alias).supports_branch_weights)
        for alias in ("dual_branch_r34", "dual_branch_resnet34"):
            self.assertEqual(registry.resolve(alias).name, "dual_branch_r34")

    def test_multiscale_alias(self):
        self.assertEqual(registry.resolve("multiscale").name, "multiscale_pyramid")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            registry.resolve("nope")

    def test_parse_models_all(self):
        specs = registry.parse_models_arg("all")
        self.assertEqual(len(specs), len(registry.MODELS))

    def test_parse_models_list_dedupes(self):
        specs = registry.parse_models_arg("baseline,resnet50,dual_branch_r50")
        self.assertEqual([s.name for s in specs], ["resnet50", "dual_branch_r50"])


class TestCheckpointResolution(unittest.TestCase):
    def test_per_model_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake = tmp_path / "fake.pth"
            fake.write_bytes(b"\x00")
            overrides = parse_overrides([f"resnet50={fake}"])
            spec = registry.resolve("baseline")
            ck = resolve_checkpoint(spec, tmp_path, overrides)
            self.assertEqual(ck.path, fake)

    def test_s3_layout_under_weights_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            d = tmp_path / "dual_branch_r50"
            d.mkdir()
            ckpt = d / "dual_branch_r50.pth"
            ckpt.write_bytes(b"\x00")
            spec = registry.resolve("dual_branch_r50")
            ck = resolve_checkpoint(spec, tmp_path)
            self.assertEqual(ck.path, ckpt)

    def test_flat_layout_under_weights_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = tmp_path / "baseline_resnet50.pth"
            ckpt.write_bytes(b"\x00")
            spec = registry.resolve("baseline")
            ck = resolve_checkpoint(spec, tmp_path)
            self.assertEqual(ck.path, ckpt)

    def test_missing_returns_none_path_with_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = registry.resolve("dual_branch")
            ck = resolve_checkpoint(spec, Path(tmp))
            self.assertIsNone(ck.path)
            self.assertGreater(len(ck.candidates_tried), 0)


class TestS3Plan(unittest.TestCase):
    def test_writes_proposal_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns = make_args(s3_plan, output=tmp, no_hash=True,
                           weights_root=str(REPO_ROOT / "decoding" / "checkpoints"),
                           results_root=str(REPO_ROOT / "decoding" / "results"),
                           bucket="s3://test-bucket/cvw")
            with redirect_stdout(io.StringIO()):
                rc = s3_plan.run(ns)
            self.assertEqual(rc, 0)
            j = Path(tmp) / "s3_structure_proposal.json"
            m = Path(tmp) / "s3_structure_proposal.md"
            self.assertTrue(j.is_file())
            self.assertTrue(m.is_file())
            payload = json.loads(j.read_text())
            self.assertEqual(payload["bucket"], "s3://test-bucket/cvw")
            self.assertIn("decoder_manifest", payload)
            self.assertEqual(
                {e["model_name"] for e in payload["decoder_manifest"]},
                set(registry.all_model_names()),
            )

    def test_runs_without_s3_access(self):
        # boto3 should never be imported by s3-plan.
        with tempfile.TemporaryDirectory() as tmp:
            ns = make_args(s3_plan, output=tmp, no_hash=True,
                           weights_root=tmp,
                           results_root=tmp,
                           bucket="s3://x/y")
            with redirect_stdout(io.StringIO()):
                rc = s3_plan.run(ns)
            self.assertEqual(rc, 0)


class TestExportSummaries(unittest.TestCase):
    def test_partial_inputs_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            results.mkdir()
            # Provide ONE expected file; everything else should be reported missing.
            (results / "robustness.json").write_text("{}")
            out = Path(tmp) / "export"
            ns = make_args(export_summaries, results_root=str(results),
                           scripts_dir=None, output=str(out),
                           extra=None, zip=False)
            with redirect_stdout(io.StringIO()):
                rc = export_summaries.run(ns)
            self.assertEqual(rc, 0)
            idx = json.loads((out / "summary_index.json").read_text())
            self.assertEqual(idx["n_collected"], 1)
            self.assertGreater(idx["n_missing"], 0)
            self.assertTrue((out / "summary_report.md").is_file())

    def test_zip_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            results.mkdir()
            (results / "robustness.json").write_text("{}")
            out = Path(tmp) / "ex"
            ns = make_args(export_summaries, results_root=str(results),
                           scripts_dir=None, output=str(out),
                           extra=None, zip=True)
            with redirect_stdout(io.StringIO()):
                export_summaries.run(ns)
            zip_path = out.with_suffix(".zip")
            self.assertTrue(zip_path.is_file())


class TestClean(unittest.TestCase):
    def test_dry_run_does_not_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.pyc").write_bytes(b"")
            (root / "keep.txt").write_text("hello")
            ns = make_args(clean_cmd, root=str(root), dry_run=True, yes=False,
                           delete_weights=False, delete_figures=False)
            with redirect_stdout(io.StringIO()):
                rc = clean_cmd.run(ns)
            self.assertEqual(rc, 0)
            self.assertTrue((root / "__pycache__").is_dir())
            self.assertTrue((root / "keep.txt").is_file())

    def test_yes_deletes_safe_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.pyc").write_bytes(b"")
            (root / ".DS_Store").write_bytes(b"")
            (root / "keep.txt").write_text("hello")
            ns = make_args(clean_cmd, root=str(root), dry_run=True, yes=True,
                           delete_weights=False, delete_figures=False)
            with redirect_stdout(io.StringIO()):
                clean_cmd.run(ns)
            self.assertFalse((root / "__pycache__").exists())
            self.assertFalse((root / ".DS_Store").exists())
            self.assertTrue((root / "keep.txt").is_file())

    def test_weights_protected_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.pth").write_bytes(b"\x00")
            ns = make_args(clean_cmd, root=str(root), dry_run=True, yes=True,
                           delete_weights=False, delete_figures=False)
            with redirect_stdout(io.StringIO()):
                clean_cmd.run(ns)
            self.assertTrue((root / "model.pth").is_file())

    def test_weights_deleted_with_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.pth").write_bytes(b"\x00")
            ns = make_args(clean_cmd, root=str(root), dry_run=True, yes=True,
                           delete_weights=True, delete_figures=False)
            with redirect_stdout(io.StringIO()):
                clean_cmd.run(ns)
            self.assertFalse((root / "model.pth").exists())


class TestTestCommandDryRun(unittest.TestCase):
    def test_dry_run_emits_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns = make_args(
                test_cmd,
                models="baseline,dual_branch_r50",
                weights_root="/nonexistent-weights",
                checkpoint=None,
                metadata=None, images=None,
                splits=str(REPO_ROOT / "decoding" / "data" / "splits.json"),
                split="test",
                batch_size=4, num_workers=0, image_size=None,
                device="cpu", threshold=0.5, limit=None,
                dual_spatial_weight=None,
                dual_spectral_weight=None,
                dual_weights="spatial=0.0,spectral=1.0",
                output=tmp, dry_run=True,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = test_cmd.run(ns)
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("resnet50", out)
            self.assertIn("dual_branch_r50", out)
            self.assertIn("MISSING", out)
            self.assertIn("spatial=0.0", out)
            self.assertIn("spectral=1.0", out)


def make_args(module, **kwargs):
    """Build an argparse.Namespace honoring the subcommand's own defaults.

    Required args without defaults are pre-seeded so parse_args succeeds; the
    caller can then override anything via kwargs.
    """
    import argparse
    parser = argparse.ArgumentParser()
    module.add_arguments(parser)
    # Pre-seed any required option with a placeholder so parse_args([]) works.
    seed: list[str] = []
    for act in parser._actions:
        if getattr(act, "required", False) and act.option_strings:
            seed += [act.option_strings[0], "PLACEHOLDER"]
    # `--model` (decode subcommand) is required and may not allow PLACEHOLDER
    # at run() time, but it parses fine here; tests override it explicitly.
    ns = parser.parse_args(seed)
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


if __name__ == "__main__":
    unittest.main()
