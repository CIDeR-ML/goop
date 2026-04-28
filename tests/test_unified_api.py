"""Unified-API tests for the merged OpticalSimulator.

Covers:
- All four cells of the {sliced, unsliced} × {stochastic, expectation} matrix
  produce sensible output types and shapes.
- Labels work in **both** stochastic and PDF modes (the new feature; PDF-mode
  labels were not supported before the refactor).
- Per-label PDF-mode sum matches the unlabeled PDF-mode total (deterministic).
- Gradients flow through the PDF path with digitization enabled (STE).
"""

from __future__ import annotations

import torch

from goop import (
    DigitizationConfig,
    OpticalSimConfig,
    OpticalSimulator,
    Response,
    SERKernel,
    ScintillationKernel,
    SlicedWaveform,
    TTSKernel,
    Waveform,
)
from goop.delays import Delays

# Reuse the synthetic photon library from the existing diff-tof tests.
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from test_differentiable_tof import _make_synth_library  # noqa: E402

DEVICE = torch.device("cpu")


def _stoch_cfg(sampler):
    return OpticalSimConfig(
        tof_sampler=sampler, delays=Delays([]),
        kernel=SERKernel(duration_ns=200.0, device=DEVICE),
        device="cpu", tick_ns=1.0, gain=-1.0,
    )


def _pdf_cfg(sampler, *, streaming=False, **overrides):
    return OpticalSimConfig(
        tof_sampler=sampler, delays=Delays([]),
        kernel=Response(
            kernels=[
                ScintillationKernel(device=DEVICE),
                TTSKernel(device=DEVICE),
                SERKernel(duration_ns=200.0, device=DEVICE),
            ],
            tick_ns=1.0, device=DEVICE,
        ),
        device="cpu", tick_ns=1.0, gain=-1.0,
        streaming=streaming, **overrides,
    )


def _inputs(n_pos: int = 5):
    pos = torch.linspace(-30, 30, n_pos).unsqueeze(-1).expand(-1, 3).contiguous()
    n_ph = torch.full((n_pos,), 100.0)
    t_step = torch.zeros(n_pos)
    return pos, n_ph, t_step


# ---------------------------------------------------------------------------
# 1. The 2x2 matrix: {sliced, unsliced} × {stochastic, expectation}
# ---------------------------------------------------------------------------


class TestMatrix:
    def test_stoch_sliced(self):
        torch.manual_seed(0)
        sim = OpticalSimulator(_stoch_cfg(_make_synth_library()))
        sw = sim.simulate(*_inputs())
        assert isinstance(sw, SlicedWaveform)
        assert sw.attrs["pe_counts"].dtype == torch.long

    def test_stoch_dense(self):
        torch.manual_seed(0)
        sim = OpticalSimulator(_stoch_cfg(_make_synth_library()))
        wf = sim.simulate(*_inputs(), sliced=False)
        assert isinstance(wf, Waveform)
        assert wf.adc.dim() == 2

    def test_pdf_sliced_streaming(self):
        sim = OpticalSimulator(_pdf_cfg(_make_synth_library(), streaming=True))
        sw = sim.simulate(*_inputs(), stochastic=False)
        assert isinstance(sw, SlicedWaveform)
        assert sw.adc.dtype == torch.float32

    def test_pdf_sliced_nonstreaming(self):
        sim = OpticalSimulator(_pdf_cfg(_make_synth_library(), streaming=False))
        sw = sim.simulate(*_inputs(), stochastic=False)
        assert isinstance(sw, SlicedWaveform)
        assert sw.adc.dtype == torch.float32

    def test_pdf_dense(self):
        sim = OpticalSimulator(_pdf_cfg(_make_synth_library(), streaming=False))
        wf = sim.simulate(*_inputs(), sliced=False, stochastic=False)
        assert isinstance(wf, Waveform)
        assert wf.adc.dtype == torch.float32


# ---------------------------------------------------------------------------
# 2. Labels work in both modes (new feature: PDF labels)
# ---------------------------------------------------------------------------


class TestLabels:
    def test_stoch_labels(self):
        torch.manual_seed(7)
        sim = OpticalSimulator(_stoch_cfg(_make_synth_library()))
        labels = torch.tensor([0, 0, 1, 1, 1])
        results = sim.simulate(*_inputs(), labels=labels)
        assert isinstance(results, list)
        assert all(isinstance(r, SlicedWaveform) for r in results)
        assert {r.attrs["label"] for r in results} <= {0, 1}

    def test_pdf_labels_streaming(self):
        sim = OpticalSimulator(_pdf_cfg(_make_synth_library(), streaming=True))
        labels = torch.tensor([0, 0, 1, 1, 1])
        results = sim.simulate(*_inputs(), labels=labels, stochastic=False)
        assert isinstance(results, list)
        assert all(isinstance(r, SlicedWaveform) for r in results)
        assert {r.attrs["label"] for r in results} <= {0, 1}

    def test_pdf_labels_nonstreaming(self):
        sim = OpticalSimulator(_pdf_cfg(_make_synth_library(), streaming=False))
        labels = torch.tensor([0, 0, 1, 1, 1])
        results = sim.simulate(*_inputs(), labels=labels, stochastic=False)
        assert isinstance(results, list)
        assert {r.attrs["label"] for r in results} <= {0, 1}

    def test_pdf_label_sum_matches_unlabeled(self):
        """Per-label PDF outputs should sum (in channel space) to the unlabeled total."""
        sim = OpticalSimulator(_pdf_cfg(_make_synth_library(), streaming=False))
        labels = torch.tensor([0, 0, 1, 1, 1])
        results = sim.simulate(*_inputs(), labels=labels, stochastic=False)
        unlabeled = sim.simulate(*_inputs(), stochastic=False)

        # Compare per-channel total PE — robust to small time-grid differences
        # between the labeled (per-label-batch) and unlabeled paths.
        per_label_pe = sum(r.attrs["pe_counts"] for r in results)
        torch.testing.assert_close(
            per_label_pe, unlabeled.attrs["pe_counts"], rtol=1e-4, atol=1e-5,
        )


# ---------------------------------------------------------------------------
# 3. Gradient flow through the unified PDF path (with digitization)
# ---------------------------------------------------------------------------


class TestPdfGradients:
    def test_grad_flows_to_pos_and_n_ph_with_digitization(self):
        cfg = _pdf_cfg(
            _make_synth_library(),
            digitization=DigitizationConfig(n_bits=14, pedestal=1500.0),
            streaming=False,
        )
        sim = OpticalSimulator(cfg)
        pos = torch.zeros(5, 3, requires_grad=True)
        n_ph = torch.full((5,), 100.0, requires_grad=True)
        t_step = torch.zeros(5)
        sw = sim.simulate(pos, n_ph, t_step, stochastic=False, add_baseline_noise=False)
        sw.adc.sum().backward()
        assert n_ph.grad is not None and torch.isfinite(n_ph.grad).all()
        assert pos.grad is not None and torch.isfinite(pos.grad).all()
        # n_ph gradient strictly nonzero (weights are linear in n_photons).
        assert (n_ph.grad.abs() > 0).all()
