"""The two ``production/run_batch.py`` input paths agree bit for bit.

* edep-sim path: edep-sim event -> jaxtpc ``process_event_light`` -> GOOP inputs
* step path:     jaxtpc step file (``load_step_event``)          -> GOOP inputs

The step file stores positions on a 0.3 mm uint16 grid and ``t0_us`` as
float16, so the edep-sim path is pushed through the same jaxtpc step writer
(``production/save.py:save_event_step``, in memory) before comparing. The
test then checks that the GOOP inputs are bitwise identical and that the
chroma waveforms simulated from them are bitwise identical under the same
seed.

Runs on the wire and pixel readouts of the doraemon_2026_1M_lartpc sample,
with the jaxtpc checkout and production settings that made it (``GOOP_JAXTPC_DIR``; the vendored ``jaxtpc/`` is an older version
whose light yield differs). Skipped when the data, jaxtpc, CUDA or
chroma-lar are missing. ``GOOP_STEP_TEST_EVENTS`` sets how many events
(default 2).
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/sdf/data/neutrino/doraemon/doraemon_2026_1M_lartpc/data")
EDEPSIM = DATA / "edepsim/j000000/edepsim_j000000-000000.h5"
# Per readout: step file (entries 0-199 = EDEPSIM events 0-199), jaxtpc detector config and
# response chunk from doraemon's production config, and the matching chroma-lar detector.
READOUTS = {
    r: dict(step=DATA / f"jaxtpc_{r}/j000000/jaxtpc_{r}_j000000-000004/step/sim_{r}_step_jaxtpc_{r}_j000000-000004.h5",
            jaxtpc_config=f"config/{cfg}.yaml", response_chunk=chunk, chroma_detector=det)
    for r, cfg, chunk, det in [
        ("wire", "cubic_wireplane_config", 28_125, "detector_config_reflect_reflect3wires"),
        ("pixel", "cubic_pixel_config", 1_000, "detector_config_pixel"),
    ]
}
JAXTPC_DIR = Path(os.environ.get("GOOP_JAXTPC_DIR", "/sdf/group/neutrino/kterao/sw/doraemon/src/JAXTPC"))
N_EVENTS = int(os.environ.get("GOOP_STEP_TEST_EVENTS", "2"))

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and importlib.util.find_spec("chroma_lar")
         and EDEPSIM.exists() and all(r["step"].exists() for r in READOUTS.values())
         and (JAXTPC_DIR / "tools").is_dir()),
    reason="needs CUDA, chroma-lar, the doraemon wire+pixel sample and its jaxtpc checkout",
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=list(READOUTS))
def env(request):
    readout = READOUTS[request.param]
    os.environ["GOOP_JAXTPC_DIR"] = str(JAXTPC_DIR)
    # same deterministic mode as `run_batch.py --deterministic` (env vars must precede cuBLAS/jax init)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()
    torch.use_deterministic_algorithms(True, warn_only=True)
    run_batch = _load("run_batch_step_vs_edepsim", REPO_ROOT / "production" / "run_batch.py")
    import tools
    if not Path(tools.__file__).resolve().is_relative_to(JAXTPC_DIR.resolve()):
        pytest.skip(f"`tools` already imported from {tools.__file__}; run this file on its own")
    from tools.geometry import generate_detector
    from tools.loader import load_event
    from tools.simulation import DetectorSimulator

    jaxtpc_save = _load("jaxtpc_step_save", JAXTPC_DIR / "production" / "save.py")
    sim = DetectorSimulator(generate_detector(str(JAXTPC_DIR / readout["jaxtpc_config"])),
                            total_pad=450_000, response_chunk_size=readout["response_chunk"],
                            include_track_hits=False)

    from goop import OpticalSimulator
    from goop.config import build_optical_config, normalize_run_config, resolve_digitization
    run_config = normalize_run_config({"sampler": {  # 10M-photon batches fit an 11 GB GPU next to jaxtpc
        "type": "chroma", "detector_config": readout["chroma_detector"], "photons_per_batch": 10_000_000}})
    goop_sim = OpticalSimulator(build_optical_config(
        run_config, gain=resolve_digitization(run_config)["gain"], n_labels=None))
    yield run_batch, sim, load_event, jaxtpc_save, goop_sim, readout["step"]
    del sim, goop_sim  # free this readout's jaxtpc + chroma GPU memory before the next one
    import gc; gc.collect(); torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)


def _edepsim_as_step(env, idx):
    """edep-sim event -> jaxtpc light -> jaxtpc step writer (in memory) -> load_step_event."""
    run_batch, sim, load_event, jaxtpc_save, *_ = env
    filled = sim.process_event_light(load_event(str(EDEPSIM), sim.config, event_idx=idx))
    buf = io.BytesIO()
    with __import__("h5py").File(buf, "w") as f:
        jaxtpc_save.save_event_step(f, "entry_000000", filled, idx, cfg=sim.config)
    return run_batch.load_step_event(buf, 0)


def _simulate(goop_sim, inputs, seed):
    torch.manual_seed(seed)
    pos, nph, tns, lbl = (torch.as_tensor(x, device="cuda") for x in inputs[:4])
    return goop_sim.simulate(pos, nph, tns, labels=lbl, stitched=True, subtract_t0=True)


@pytest.mark.parametrize("idx", range(N_EVENTS))
def test_step_file_matches_edepsim_bitwise(env, idx):
    run_batch, *_, goop_sim, step = env
    from_step = run_batch.load_step_event(str(step), idx)
    from_edep = _edepsim_as_step(env, idx)

    # GOOP inputs: pos, n_photons, t_step_ns, labels, (pdg=None), de, n_segments
    assert from_step[-1] == from_edep[-1]
    for name, a, b in zip(["pos", "n_photons", "t_ns", "labels"], from_step, from_edep):
        assert a.dtype == b.dtype and np.array_equal(a, b), f"event {idx}: {name} differs"
    assert np.array_equal(from_step[5], from_edep[5]), f"event {idx}: de differs"

    # chroma waveforms from identical inputs and seed
    wf_step = _simulate(goop_sim, from_step, seed=42 + idx)
    wf_edep = _simulate(goop_sim, from_edep, seed=42 + idx)
    assert len(wf_step) == len(wf_edep) == 2  # one per TPC volume
    for a, b in zip(wf_step, wf_edep):
        assert a.attrs["label"] == b.attrs["label"]
        assert int(a.attrs["pe_counts"].sum()) > 0
        for field in ("adc", "offsets", "t0_ns", "pmt_id"):
            assert torch.equal(getattr(a, field), getattr(b, field)), f"event {idx}: {field} differs"
        assert torch.equal(a.attrs["pe_counts"], b.attrs["pe_counts"])
