"""Smoke test for ``production/run_batch.py``.

Catches the kind of merge artifacts that pytest-without-this-file lets through:
  * undefined-name bugs in print format strings (``t_vox``, ``n_after``, ...)
  * missing imports (``create_siren_tof_sampler``)
  * stale references to removed kwargs (``do_voxelize``)

Two layers:
  1. ``test_imports``: load ``production/run_batch.py`` as a module — checks
     all imports resolve and module-level code is well-formed. Catches
     missing-import / NameError bugs without any GPU.
  2. ``test_main_subprocess``: run ``run_batch.main()`` as a subprocess on the
     real ``out.h5`` fixture if it exists. Catches print-format-string bugs
     and any other runtime issues that only surface on a real event loop.
     Skipped automatically when the fixture or a GPU isn't available.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_BATCH = REPO_ROOT / "production" / "run_batch.py"
DEFAULT_FIXTURE_H5 = "/sdf/home/y/youngsam/sw/dune/sirentv/data/out.h5"


@pytest.fixture
def out_h5_path():
    """Resolves the real out.h5 fixture; skips the test if not on this cluster."""
    p = os.environ.get("GOOP_TEST_OUT_H5", DEFAULT_FIXTURE_H5)
    if not Path(p).exists():
        pytest.skip(f"out.h5 fixture not found at {p}")
    return p


def test_imports():
    """``run_batch.py`` should load as a module without exceptions.

    Catches:
      * missing imports (``create_siren_tof_sampler`` was the latest one),
      * stale kwargs in argparse / function bodies that no longer exist
        upstream (``do_voxelize`` / ``cfg.voxel_size_mm``),
      * syntax breakage from merge artifacts.
    """
    spec = importlib.util.spec_from_file_location("run_batch_smoke_mod", str(RUN_BATCH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Module-level code lives inside `def main():`, so loading it doesn't
    # parse argv or hit GPU. We just need to know it imports cleanly.
    spec.loader.exec_module(module)
    # Spot-checks: helpers production callers depend on must be present.
    assert hasattr(module, "extract_goop_inputs")
    assert hasattr(module, "voxelize_labeled")
    assert hasattr(module, "main")
    # Distribution sentinels imported by the script must resolve.
    assert hasattr(module, "Poisson")
    assert hasattr(module, "Uniform")
    assert hasattr(module, "HalfNormal")


def test_run_config_yaml_defaults(tmp_path):
    """YAML run configs should populate argparse defaults and remain overrideable."""
    spec = importlib.util.spec_from_file_location("run_batch_smoke_mod", str(RUN_BATCH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_config = tmp_path / "run.yml"
    run_config.write_text(
        "\n".join([
            "data: /tmp/input.h5",
            "detector_config: jaxtpc/config/cubic_wireplane_config.yaml",
            "dataset: yaml_dataset",
            "outdir: /tmp/out",
            "events: null",
            "events_per_file: 7",
            "digitize: false",
            "dark_noise: true",
            "align: true",
            "plib_path: /tmp/plib.h5",
            "voxel_dx: 10.0",
        ]),
        encoding="utf-8",
    )

    args = module.parse_args([
        "--run-config", str(run_config),
        "--events", "3",
        "--digitize",
        "--no-dark-noise",
        "--no-align",
    ])

    assert args.run_config == str(run_config)
    assert args.data == "/tmp/input.h5"
    assert args.config == "jaxtpc/config/cubic_wireplane_config.yaml"
    assert args.dataset == "yaml_dataset"
    assert args.events == 3
    assert args.events_per_file == 7
    assert args.no_digitize is False
    assert args.dark_noise is False
    assert args.align is False
    assert args.pca_lut_path == "/tmp/plib.h5"
    assert args.voxel_dx == 10.0


@pytest.mark.skipif(
    shutil.which("python") is None,
    reason="No python available for subprocess smoke",
)
def test_main_subprocess_one_event(out_h5_path, tmp_path):
    """End-to-end production smoke: 1 event, LUT, --voxel-dx 10, interaction labels.

    This is the only test that catches print-format-string bugs (``t_vox``,
    ``n_after``) and other run-time issues introduced by merges. Skipped
    when out.h5 isn't available (so the suite still passes off-cluster).
    """
    import torch
    if not torch.cuda.is_available():
        pytest.skip("smoke test needs a GPU")
    free_mb = (torch.cuda.mem_get_info()[0] // (1024 * 1024)
               if hasattr(torch.cuda, "mem_get_info") else 0)
    if free_mb < 28_000:
        pytest.skip(f"need ~28 GB free GPU for LUT-eager; have {free_mb} MB")

    out = tmp_path / "out"
    out.mkdir()

    cmd = [
        sys.executable, str(RUN_BATCH),
        "--data", out_h5_path,
        "--config", str(REPO_ROOT / "jaxtpc" / "config" / "cubic_wireplane_config.yaml"),
        "--dataset", "smoke",
        "--outdir", str(out),
        "--events", "1",
        "--events-per-file", "1",
        "--workers", "0",
        "--label-key", "interaction",
        "--sampler", "lut",
        "--voxel-dx", "10",
    ]
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0, (
        f"run_batch.main() failed:\n"
        f"--- stdout (tail) ---\n{res.stdout[-2000:]}\n"
        f"--- stderr (tail) ---\n{res.stderr[-2000:]}"
    )
    # Output file should exist and be non-empty.
    opticals = list((out / "sensor_optical").glob("*.h5"))
    assert len(opticals) == 1, f"expected one optical file, got {opticals}"
    assert opticals[0].stat().st_size > 0
    # Print sanity: the per-event timing row must include the t_vox column.
    assert "t_vox" in res.stdout
    assert "Average:" in res.stdout
