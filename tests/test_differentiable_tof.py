"""Tests for DifferentiableTOFSampler — Phase 2 of the differentiable refactor.

The diff TOF replaces stochastic per-photon sampling with deterministic
PDF deposition: ``expected = v · n_photons · pmt_qe`` (no Poisson) and the
PCA-reconstructed quantile function gives the per-bin probability mass.
Output is synthetic ``(times, channels, weights)`` triples that, when
histogrammed with those weights, reproduce the per-PMT expected PDF.

A synthetic photon library is constructed via ``TOFSampler.from_arrays``
so these tests run without DEFAULT_PLIB_PATH.
"""

from __future__ import annotations

import pytest
import torch

from goop import (
    DifferentiableTOFSampler,
    OpticalSimConfig,
    OpticalSimulator,
    Response,
    SERKernel,
    TOFSampler,
)
from goop.delays import Delays

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Synthetic photon library
# ---------------------------------------------------------------------------


def _make_synth_library(
    cls=TOFSampler,
    n_pmts: int = 4,
    nx: int = 5, ny: int = 5, nz: int = 5,
    K: int = 3, Q: int = 20,
    n_simulated: float = 100.0,
    pmt_qe: float = 0.5,
    device: torch.device = DEVICE,
) -> TOFSampler:
    """Tiny in-memory photon library with predictable PDFs.

    - Voxel grid 5×5×5 in a [-50, 50] mm cube
    - n_pmts (per side) at fixed positions; visibility falls off ~1/r²
    - PCA basis is trivial (zero coefficients): every pair has the same
      quantile shape ``pca_mean`` shifted by its per-pair ``t0``
    - mode='linear' so quantile reconstruction is just `raw + t0` (no log)
    """
    n_voxels = nx * ny * nz
    # Build a smooth visibility surrogate: brighter for voxels near each PMT
    voxel_xyz = torch.stack(torch.meshgrid(
        torch.linspace(-50, 50, nx),
        torch.linspace(-50, 50, ny),
        torch.linspace(-50, 50, nz),
        indexing="ij",
    ), dim=-1).reshape(-1, 3)  # (V, 3)

    # PMT layout: along the +x cathode face, evenly spaced in y
    pmt_xyz = torch.stack([
        torch.zeros(n_pmts),
        torch.linspace(-30, 30, n_pmts),
        torch.zeros(n_pmts),
    ], dim=-1)  # (P, 3)

    # vis[v, p] = 1 / (distance² + 1)
    diff = voxel_xyz.unsqueeze(1) - pmt_xyz.unsqueeze(0)  # (V, P, 3)
    dist2 = diff.pow(2).sum(-1)                            # (V, P)
    vis = 5.0 / (dist2 / 100.0 + 1.0)                      # (V, P) ~ 0.1 to 5
    t0 = dist2.sqrt() * 0.05                               # (V, P) ns

    coeffs = torch.zeros(n_voxels, n_pmts, K)              # trivial PCA basis
    pca_mean = torch.linspace(0.5, 10.0, Q)                # (Q,) quantile times rel. to t0
    pca_components = torch.zeros(K, Q)
    u_grid = torch.linspace(0.0, 1.0, Q + 1)[1:]           # (Q,) uniform-like grid

    return cls.from_arrays(
        vis=vis, t0=t0, coeffs=coeffs,
        pca_mean=pca_mean, pca_components=pca_components, u_grid=u_grid,
        numvox=torch.tensor([nx, ny, nz], dtype=torch.long),
        min_xyz=torch.tensor([-50.0, -50.0, -50.0], dtype=torch.float64),
        max_xyz=torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64),
        log_quantile_C=0.0, t_max_ns=20.0, mode="linear",
        n_simulated=n_simulated, device=str(device), interpolate=True,
        pmt_qe=pmt_qe,
    )


# ---------------------------------------------------------------------------
# 1. from_arrays smoke test on the existing TOFSampler
# ---------------------------------------------------------------------------


class TestFromArrays:
    def test_construct_and_basic_attrs(self):
        s = _make_synth_library()
        assert isinstance(s, TOFSampler)
        assert s.n_channels == 8  # 4 PMTs × 2 sides
        assert s.vis.shape == (125, 4)
        assert s.coeffs.shape == (125, 4, 3)
        assert s.u_grid.shape == (20,)

    def test_existing_sample_still_works(self):
        torch.manual_seed(0)
        s = _make_synth_library()
        pos = torch.zeros(10, 3)
        n_ph = torch.full((10,), 100)
        t_step = torch.zeros(10)
        times, channels, weights, source_idx = s.sample_photons(
            pos, n_ph, t_step=t_step, return_source_idx=True,
        )
        assert times.dim() == 1
        assert channels.dim() == 1
        assert weights.dim() == 1
        assert source_idx.dim() == 1
        assert times.numel() == channels.numel() == weights.numel() == source_idx.numel()


# ---------------------------------------------------------------------------
# 2. Yield equivalence: diff PDF integral ≈ stochastic Poisson mean
# ---------------------------------------------------------------------------


class TestYieldEquivalence:
    """Sum of diff weights per channel should equal averaged stoch count
    per channel — this is the law of large numbers applied to Poisson
    photon counts at the per-PMT level."""

    @pytest.mark.parametrize("n_pos,n_ph_each,n_runs,tol_rel", [(50, 1000, 300, 0.05)])
    def test_per_channel_yield(self, n_pos, n_ph_each, n_runs, tol_rel):
        torch.manual_seed(0)
        diff_sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        stoch_sampler = _make_synth_library(cls=TOFSampler)

        pos = torch.linspace(-40, 40, n_pos).unsqueeze(-1).expand(-1, 3).contiguous()
        n_ph = torch.full((n_pos,), n_ph_each)
        t_step = torch.zeros(n_pos)

        # Diff: deterministic PDF integral per channel
        d_times, d_channels, d_weights = diff_sampler.sample_photons(pos, n_ph, t_step, stochastic=False)
        diff_per_ch = torch.zeros(diff_sampler.n_channels)
        diff_per_ch.scatter_add_(0, d_channels.long(), d_weights)

        # Stoch: average count per channel over n_runs
        stoch_accum = torch.zeros(stoch_sampler.n_channels)
        for _ in range(n_runs):
            _, s_channels, _ = stoch_sampler.sample_photons(pos, n_ph, t_step=t_step)
            stoch_accum.scatter_add_(
                0, s_channels.long(), torch.ones_like(s_channels, dtype=torch.float32)
            )
        stoch_mean = stoch_accum / n_runs

        # Compare per channel.  Skip channels whose expected count per run is
        # too low for the shot-noise floor to be small (rel std ≈ 1/√(N·d)).
        for ch in range(diff_sampler.n_channels):
            d, s = diff_per_ch[ch].item(), stoch_mean[ch].item()
            if d < 5.0:
                continue
            assert abs(d - s) / d < tol_rel, (
                f"ch {ch}: diff={d:.3f}, stoch_mean={s:.3f}, "
                f"rel diff = {abs(d-s)/d:.4f}"
            )


# ---------------------------------------------------------------------------
# 3. Per-bin histogram equivalence
# ---------------------------------------------------------------------------


class TestHistogramEquivalence:
    def test_per_channel_histogram(self):
        torch.manual_seed(1)
        diff_sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        stoch_sampler = _make_synth_library(cls=TOFSampler)

        n_pos = 30
        pos = torch.linspace(-30, 30, n_pos).unsqueeze(-1).expand(-1, 3).contiguous()
        n_ph = torch.full((n_pos,), 1000)
        t_step = torch.zeros(n_pos)

        tick = 0.5
        n_bins = 50  # covers 25 ns; PDFs span ~0.5–10 ns + t0 (small)
        n_ch = diff_sampler.n_channels

        def hist(times, channels, weights):
            out = torch.zeros(n_ch, n_bins)
            if times.numel() == 0:
                return out
            bin_idx = (times / tick).long().clamp(0, n_bins - 1)
            flat = channels.long() * n_bins + bin_idx
            out.view(-1).scatter_add_(0, flat, weights)
            return out

        d_times, d_channels, d_weights = diff_sampler.sample_photons(pos, n_ph, t_step, stochastic=False)
        diff_hist = hist(d_times, d_channels, d_weights)

        n_runs = 600
        stoch_hist = torch.zeros(n_ch, n_bins)
        for _ in range(n_runs):
            t_s, c_s, _ = stoch_sampler.sample_photons(pos, n_ph, t_step=t_step)
            stoch_hist += hist(t_s, c_s, torch.ones_like(t_s))
        stoch_hist /= n_runs

        # Where signal is present, per-bin diff should match stoch within shot
        # noise.  Per-bin shot-noise dominates for low-expected bins; the
        # yield-equivalence test above is the strict correctness check.
        for ch in range(n_ch):
            sig_l2 = diff_hist[ch].norm().item()
            if sig_l2 < 0.5:
                continue
            res_l2 = (diff_hist[ch] - stoch_hist[ch]).norm().item()
            assert res_l2 / sig_l2 < 0.15, (
                f"ch {ch}: per-bin residual {res_l2/sig_l2:.4f} > 0.15"
            )


# ---------------------------------------------------------------------------
# 4. Gradient flow through n_photons (and pos via trilinear interp)
# ---------------------------------------------------------------------------


class TestGradients:
    def test_grad_flows_to_n_photons(self):
        sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        pos = torch.zeros(5, 3)
        t_step = torch.zeros(5)
        n_ph = torch.full((5,), 100.0, requires_grad=True)

        times, channels, weights = sampler.sample_photons(pos, n_ph, t_step, stochastic=False)
        loss = weights.sum()
        loss.backward()

        assert n_ph.grad is not None
        assert torch.isfinite(n_ph.grad).all()
        assert (n_ph.grad > 0).all(), (
            "weights are linear in n_photons → gradient should be strictly positive"
        )

    def test_grad_flows_to_pos(self):
        sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        # positions where vis is meaningful (interior of voxel grid)
        pos = torch.tensor([[10.0, 5.0, -5.0], [-10.0, 0.0, 10.0]], requires_grad=True)
        t_step = torch.zeros(2)
        n_ph = torch.full((2,), 100.0)

        _, _, weights = sampler.sample_photons(pos, n_ph, t_step, stochastic=False)
        loss = weights.sum()
        loss.backward()

        assert pos.grad is not None
        assert torch.isfinite(pos.grad).all()
        # at least one pos coord should have non-trivial gradient (vis varies w/ position)
        assert pos.grad.abs().max().item() > 1e-9


# ---------------------------------------------------------------------------
# 5. Integration: full diff pipeline with sample_pdf
# ---------------------------------------------------------------------------


class TestDiffSimIntegration:
    def test_pipeline_runs_with_sample_pdf(self):
        sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        ser = SERKernel(duration_ns=200.0, device=DEVICE)
        cfg = OpticalSimConfig(
            tof_sampler=sampler, delays=Delays([]),
            kernel=Response(kernels=[ser], tick_ns=1.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-1.0,
        )
        sim = OpticalSimulator(cfg)

        pos = torch.zeros(5, 3)
        n_ph = torch.full((5,), 100)
        t_step = torch.zeros(5)
        sw = sim.simulate(pos, n_ph, t_step, sliced=True, stochastic=False, add_baseline_noise=False)

        from goop import SlicedWaveform
        assert isinstance(sw, SlicedWaveform)
        assert sw.n_channels == sampler.n_channels
        # pe_counts should be float (weighted sum), not int from bincount
        assert sw.attrs["pe_counts"].dtype == torch.float32

    def test_pipeline_grad_through_n_photons(self):
        sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        ser = SERKernel(duration_ns=200.0, device=DEVICE)
        cfg = OpticalSimConfig(
            tof_sampler=sampler, delays=Delays([]),
            kernel=Response(kernels=[ser], tick_ns=1.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-1.0,
        )
        sim = OpticalSimulator(cfg)

        pos = torch.zeros(5, 3)
        n_ph = torch.full((5,), 100.0, requires_grad=True)
        t_step = torch.zeros(5)
        sw = sim.simulate(pos, n_ph, t_step, sliced=True, stochastic=False, add_baseline_noise=False)
        loss = sw.adc.pow(2).sum()
        loss.backward()

        assert n_ph.grad is not None
        assert torch.isfinite(n_ph.grad).all()
        assert n_ph.grad.abs().max().item() > 0

    def test_digitization_ste_grad_through_n_photons(self):
        """With STE digitization: forward quantizes, backward still reaches n_photons."""
        from goop import DigitizationConfig
        sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        ser = SERKernel(duration_ns=200.0, device=DEVICE)
        cfg = OpticalSimConfig(
            tof_sampler=sampler, delays=Delays([]),
            kernel=Response(kernels=[ser], tick_ns=1.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-1.0,
            digitization=DigitizationConfig(n_bits=14, pedestal=1500.0),
        )
        sim = OpticalSimulator(cfg)

        pos = torch.zeros(5, 3)
        n_ph = torch.full((5,), 100.0, requires_grad=True)
        t_step = torch.zeros(5)
        sw = sim.simulate(pos, n_ph, t_step, sliced=True, stochastic=False, add_baseline_noise=False)

        # Forward is quantized
        assert torch.equal(sw.adc, sw.adc.round())
        assert sw.adc.min().item() >= 0
        assert sw.adc.max().item() <= (1 << 14) - 1

        # Backward: STE lets gradient reach n_photons even through round+clamp
        loss = sw.adc.pow(2).sum()
        loss.backward()
        assert n_ph.grad is not None
        assert torch.isfinite(n_ph.grad).all()
        assert n_ph.grad.abs().max().item() > 0

    def test_pe_counts_grad_through_n_photons(self):
        """Loss on pe_counts (sidecar) should also backprop to n_photons."""
        sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        ser = SERKernel(duration_ns=200.0, device=DEVICE)
        cfg = OpticalSimConfig(
            tof_sampler=sampler, delays=Delays([]),
            kernel=Response(kernels=[ser], tick_ns=1.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-1.0,
        )
        sim = OpticalSimulator(cfg)

        pos = torch.zeros(5, 3)
        n_ph = torch.full((5,), 100.0, requires_grad=True)
        t_step = torch.zeros(5)
        sw = sim.simulate(pos, n_ph, t_step, sliced=True, stochastic=False, add_baseline_noise=False)
        loss = sw.attrs["pe_counts"].sum()
        loss.backward()

        assert n_ph.grad is not None
        assert torch.isfinite(n_ph.grad).all()
        assert n_ph.grad.abs().max().item() > 0

    def test_baseline_noise_works_with_pdf_path(self):
        """baseline_noise_std > 0 is fine even with sample_pdf — purely additive."""
        torch.manual_seed(0)
        sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        ser = SERKernel(duration_ns=200.0, device=DEVICE)
        cfg = OpticalSimConfig(
            tof_sampler=sampler, delays=Delays([]),
            kernel=Response(kernels=[ser], tick_ns=1.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-1.0,
            baseline_noise_std=3.0,
        )
        sim = OpticalSimulator(cfg)
        sw = sim.simulate(
            torch.zeros(5, 3), torch.full((5,), 100), torch.zeros(5),
            sliced=True, stochastic=False, add_baseline_noise=True,
        )
        from goop import SlicedWaveform
        assert isinstance(sw, SlicedWaveform)
        # noise is on the order of baseline_noise_std
        assert sw.adc.std().item() > 1.0


# ---------------------------------------------------------------------------
# 6. Streaming-histogram path: same math as sample_pdf, smaller peak memory
# ---------------------------------------------------------------------------


class TestStreamingHistogram:
    """``cfg.streaming=True`` must produce the same ADC content as the default
    ``sample_pdf`` + ``from_photons`` path on identical inputs.
    """

    def _make_cfgs(self, sampler, *, baseline_noise_std=0.0, pos_batch_size=7):
        """Build paired configs identical except for the ``streaming`` flag."""
        ser = SERKernel(duration_ns=200.0, device=DEVICE)

        def _build(streaming):
            return OpticalSimConfig(
                tof_sampler=sampler, delays=Delays([]),
                kernel=Response(kernels=[ser], tick_ns=1.0, device=DEVICE),
                device="cpu", tick_ns=1.0, gain=-1.0,
                baseline_noise_std=baseline_noise_std,
                streaming=streaming,
                pos_batch_size=pos_batch_size,
            )
        return _build(False), _build(True)

    @staticmethod
    def _align(adc, t0, tick_ns, t_start, n_bins):
        """Pad ``adc`` onto the (t_start, n_bins) reference grid."""
        import torch.nn.functional as F
        n_ch = adc.shape[0]
        offset = int(round((t0 - t_start) / tick_ns))
        out = torch.zeros(n_ch, n_bins, device=adc.device, dtype=adc.dtype)
        src_lo = max(0, -offset)
        dst_lo = max(0, offset)
        n_copy = min(adc.shape[1] - src_lo, n_bins - dst_lo)
        if n_copy > 0:
            out[:, dst_lo:dst_lo + n_copy] = adc[:, src_lo:src_lo + n_copy]
        return out

    def test_streaming_matches_sample_pdf(self):
        """ADC output must match between ``streaming=False`` and ``streaming=True``."""
        sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        cfg_def, cfg_str = self._make_cfgs(sampler)

        sim_def = OpticalSimulator(cfg_def)
        sim_str = OpticalSimulator(cfg_str)

        torch.manual_seed(0)
        pos = torch.linspace(-30, 30, 20).unsqueeze(-1).expand(-1, 3).contiguous()
        n_ph = torch.full((20,), 200.0)
        t_step = torch.linspace(0.0, 5.0, 20)

        sw_def = sim_def.simulate(pos, n_ph, t_step, sliced=True, stochastic=False, add_baseline_noise=False)
        wf_def = sw_def.deslice()
        wf_str = sim_str.simulate(pos, n_ph, t_step, sliced=True, stochastic=False, add_baseline_noise=False).deslice()

        # Align both onto a common time grid spanning the union.
        tick = wf_def.tick_ns
        assert abs(wf_str.tick_ns - tick) < 1e-9
        t_start = min(wf_def.t0, wf_str.t0)
        n_bins = max(
            int(round((wf_def.t0 - t_start) / tick)) + wf_def.adc.shape[1],
            int(round((wf_str.t0 - t_start) / tick)) + wf_str.adc.shape[1],
        )
        adc_def = self._align(wf_def.adc, wf_def.t0, tick, t_start, n_bins)
        adc_str = self._align(wf_str.adc, wf_str.t0, tick, t_start, n_bins)

        diff = (adc_def - adc_str).abs()
        scale = adc_def.abs().max().clamp(min=1e-6)
        rel_err = (diff.max() / scale).item()
        # Should be essentially identical up to FFT/scatter float noise.
        assert rel_err < 1e-3, f"streaming adc diverges from sample_pdf adc by rel_err={rel_err}"

    def test_streaming_grad_through_n_photons(self):
        """Backward through ``streaming=True`` reaches ``n_photons``."""
        sampler = _make_synth_library(cls=DifferentiableTOFSampler)
        _, cfg_str = self._make_cfgs(sampler)
        sim = OpticalSimulator(cfg_str)

        pos = torch.zeros(10, 3)
        n_ph = torch.full((10,), 100.0, requires_grad=True)
        t_step = torch.zeros(10)
        wf = sim.simulate(pos, n_ph, t_step, sliced=True, stochastic=False, add_baseline_noise=False)
        loss = wf.adc.pow(2).sum()
        loss.backward()

        assert n_ph.grad is not None
        assert torch.isfinite(n_ph.grad).all()
        assert n_ph.grad.abs().max().item() > 0
