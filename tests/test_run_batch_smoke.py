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
            "start_event: 20",
            "events: null",
            "events_per_file: 7",
            "file_index_offset: 3",
            "skip_existing: true",
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
        "--start-event", "8",
        "--events", "3",
        "--file-index-offset", "9",
        "--no-skip-existing",
        "--digitize",
        "--no-dark-noise",
        "--no-align",
    ])

    assert args.run_config == str(run_config)
    assert args.data == "/tmp/input.h5"
    assert args.config == "jaxtpc/config/cubic_wireplane_config.yaml"
    assert args.dataset == "yaml_dataset"
    assert args.start_event == 8
    assert args.events == 3
    assert args.events_per_file == 7
    assert args.file_index_offset == 9
    assert args.skip_existing is False
    assert args.no_digitize is False
    assert args.dark_noise is False
    assert args.align is False
    assert args.pca_lut_path == "/tmp/plib.h5"
    assert args.voxel_dx == 10.0


def test_sampler_chroma_replaces_yaml_lut(tmp_path):
    """``--sampler chroma`` drops the YAML's photon-library keys and takes the seeded chroma defaults."""
    spec = importlib.util.spec_from_file_location("run_batch_smoke_mod", str(RUN_BATCH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_config = tmp_path / "run.yml"
    run_config.write_text("sampler:\n  type: lut\n  plib_path: /tmp/plib.h5\n  lazy: true\n", encoding="utf-8")

    args = module.parse_args(["--run-config", str(run_config), "--sampler", "chroma", "--pmt-qe", "0.2"])
    assert args.resolved_config["sampler"] == {
        "type": "chroma",
        "detector_config": "detector_config_reflect_reflect3wires",
        "wavelength": 450.0,
        "pmt_qe": 0.2,
        "seed": 0,
        "photons_per_batch": 50_000_000,
        "max_steps": 1000,
        "device": "cuda:0",
    }


@pytest.mark.skipif(
    shutil.which("python") is None,
    reason="No python available for subprocess smoke",
)
@pytest.mark.parametrize("sampler, min_free_mb", [("lut", 28_000), ("chroma", 9_000)])
def test_main_subprocess_one_event(out_h5_path, tmp_path, sampler, min_free_mb):
    """End-to-end production smoke: 1 event per sampler, --voxel-dx 10, interaction labels.

    This is the only test that catches print-format-string bugs (``t_vox``,
    ``n_after``) and other run-time issues introduced by merges. Skipped
    when out.h5 isn't available (so the suite still passes off-cluster).
    """
    import torch
    if not torch.cuda.is_available():
        pytest.skip("smoke test needs a GPU")
    free_mb = (torch.cuda.mem_get_info()[0] // (1024 * 1024)
               if hasattr(torch.cuda, "mem_get_info") else 0)
    if free_mb < min_free_mb:
        pytest.skip(f"need ~{min_free_mb // 1000} GB free GPU for {sampler}; have {free_mb} MB")

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
        "--sampler", sampler,
        "--voxel-dx", "10",
    ]
    if sampler == "chroma":
        # smaller TriChroma batches than the default 50M, so the event fits on 11 GB GPUs
        run_config = tmp_path / "run.yml"
        run_config.write_text("sampler:\n  type: chroma\n  photons_per_batch: 5000000\n", encoding="utf-8")
        cmd += ["--run-config", str(run_config)]
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
    import h5py
    with h5py.File(opticals[0], "r") as f:
        assert f["config"].attrs["sampler_type"] == sampler
    # Print sanity: the per-event timing row must include the t_vox column.
    assert "t_vox" in res.stdout
    assert "Average:" in res.stdout


def test_load_step_event(tmp_path):
    """A jaxtpc step file decodes to global mm, ceil'd photons, ns times and volume labels."""
    import h5py
    import numpy as np

    spec = importlib.util.spec_from_file_location("run_batch_smoke_mod", str(RUN_BATCH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    path = tmp_path / "step.h5"
    with h5py.File(path, "w") as f:
        evt = f.create_group("entry_000000")
        evt.attrs["n_volumes"] = 2
        empty = evt.create_group("volume_0")
        empty.attrs["n_actual"] = 0
        vg = evt.create_group("volume_1")
        vg.attrs.update(n_actual=2, pos_step_mm=0.3,
                        pos_origin_x=10.0, pos_origin_y=-20.0, pos_origin_z=5.0)
        vg["positions"] = np.array([[0, 0, 0], [10, 20, 30]], dtype=np.uint16)
        vg["photons"] = np.array([1.2, 3.0], dtype=np.float32)
        vg["t0_us"] = np.array([0.5, 2.0], dtype=np.float16)
        vg["de"] = np.array([0.1, 0.2], dtype=np.float16)

    assert module.is_step_file(str(path))
    assert module.get_num_events(str(path)) == 1
    pos, nph, tns, lbl, pdg, de, n = module.load_step_event(str(path), 0)
    np.testing.assert_allclose(pos, [[10, -20, 5], [13, -14, 14]], atol=1e-5)
    np.testing.assert_array_equal(nph, [2, 3])
    np.testing.assert_allclose(tns, [500.0, 2000.0])
    np.testing.assert_array_equal(lbl, [1, 1])
    assert pdg is None and n == 2 and de.shape == (2,)
