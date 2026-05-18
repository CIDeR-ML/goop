"""Differentiable optical-simulation pipeline.

Replaces stochastic per-photon delay sampling with deterministic convolution
against a composite `Response` kernel that includes the delay PDFs, and
replaces stochastic per-photon TOF sampling with a deterministic PDF
deposition via `DifferentiableTOFSampler.sample_pdf`.

Required config
---------------
- `cfg.kernel` is a `Response` (or any `ConvolutionKernelBase`) that
  already encodes the delay PDFs the user wants — typically
  `create_default_response()`.
- `cfg.tof_sampler` exposes a `sample_pdf(...)` method
  (e.g. `DifferentiableTOFSampler`).
"""

from __future__ import annotations

import math
from typing import Any, List, Union

import torch
import torch.nn.functional as F

from .digitize import digitize_ste
from .simulator import OpticalSimConfig, OpticalSimulator
from .waveform import SlicedWaveform, Waveform
from .waveform_utils import _next_fft_size

ArrayLike = Union[torch.Tensor, Any]

# Padding constants for V2 sparse streaming time-grouping.
_GAP_THRESHOLD_PAD_NS = 100.0  # safety margin so a gap-split can't bisect a kernel-extent tail
_N_BINS_GROUP_PAD = 10         # extra bins so late photons aren't truncated by integer rounding


class _HistConvFunction(torch.autograd.Function):
    """Fused hard-bin scatter + FFT convolve, with smooth analytic backward.

    Forward
    -------
        hist[c, k]  = scatter_add (weights, by floor(q_abs/tick - t0_ref), by pmt)
        pred[c, t]  = gain · (hist ⋆ kernel)[c, t]   (FFT-based linear convolution)

    Backward (custom — does NOT differentiate the floor cast)
    ---------------------------------------------------------
        grad_pred -> grad_pred_f via rFFT
        grad_hist_f = grad_pred_f · conj(kernel_f)               # standard adjoint
        score_f     = grad_pred_f · conj(kernel_deriv_f)         # = -2πi·f · grad_hist_f
        Then irFFT and crop to (n_ch, n_bins).

        For each saved (m, q) pair, evaluate `grad_hist` and `score` at the
        continuous arrival time `q_abs[m, q]` via cubic B-spline interpolation
        (4-tap, C^2 smooth).  This is what `weights[m, q]` and `q_abs[m, q]`
        would have received if forward had soft-binned with a sinc-equivalent
        kernel.

        dL/d(weights[m,q]) =  gain · grad_hist( q_abs[m,q] )
        dL/d(q_abs [m,q]) = -gain · weights[m,q] · score( q_abs[m,q] )

    `pmt_idx`, `kernel_f`, `kernel_deriv_f`, and the bookkeeping scalars receive
    no gradient (kernel_f / kernel_deriv_f are pre-computed and treated as
    constants here).
    """

    @staticmethod
    def forward(
        ctx,
        q_abs,            # (M, Q) float — autograd-live arrival times in ns
        weights,          # (M, Q) float — autograd-live per-pair weights
        pmt_idx,          # (M,)   long  — channel id per pair
        kernel_f,         # ((n_fft//2)+1,) complex — pre-computed rfft of kernel
        kernel_deriv_f,   # same shape — i·2π·f · kernel_f
        t0_ref,           # python float
        tick_ns,          # python float
        n_bins,           # python int — histogram length
        n_ch,             # python int
        gain,             # python float
        kernel_extent_bins,  # python int — kernel.shape[0]
    ):
        device = q_abs.device
        n_fft = (kernel_f.shape[0] - 1) * 2

        # Hard-bin scatter into a fresh histogram.
        bin_idx = ((q_abs.detach() - t0_ref) / tick_ns).floor().long()  # (M, Q)
        in_w = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx_c = bin_idx.clamp(0, n_bins - 1)
        flat_idx = pmt_idx.unsqueeze(-1) * n_bins + bin_idx_c           # (M, Q)
        w_in = (weights.detach() * in_w).reshape(-1)
        hist = torch.zeros(n_ch, n_bins, device=device, dtype=torch.float32)
        hist.view(-1).scatter_add_(0, flat_idx.reshape(-1), w_in)

        # FFT convolve hist with kernel.
        padded = F.pad(hist, (0, n_fft - n_bins))
        hist_f = torch.fft.rfft(padded, dim=-1)
        pred_f = hist_f * kernel_f.unsqueeze(0)
        out_len = n_bins + kernel_extent_bins - 1
        pred = gain * torch.fft.irfft(pred_f, n=n_fft, dim=-1)[:, :out_len]

        ctx.save_for_backward(q_abs, weights, pmt_idx, kernel_f, kernel_deriv_f)
        ctx.t0_ref = t0_ref
        ctx.tick_ns = tick_ns
        ctx.n_bins = n_bins
        ctx.n_ch = n_ch
        ctx.gain = gain
        ctx.n_fft = n_fft
        ctx.out_len = out_len
        return pred

    @staticmethod
    def backward(ctx, grad_pred):
        q_abs, weights, pmt_idx, kernel_f, kernel_deriv_f = ctx.saved_tensors
        t0_ref, tick_ns = ctx.t0_ref, ctx.tick_ns
        n_bins, n_ch, gain = ctx.n_bins, ctx.n_ch, ctx.gain
        n_fft, out_len = ctx.n_fft, ctx.out_len
        device = grad_pred.device

        # Pad upstream gradient and rFFT.
        grad_pred_padded = F.pad(grad_pred, (0, n_fft - out_len))
        grad_pred_f = torch.fft.rfft(grad_pred_padded, dim=-1)

        # Two adjoint convolutions in Fourier — share the same FFT of grad_pred.
        grad_hist_f = grad_pred_f * kernel_f.conj().unsqueeze(0)
        score_f     = grad_pred_f * kernel_deriv_f.conj().unsqueeze(0)
        grad_hist = torch.fft.irfft(grad_hist_f, n=n_fft, dim=-1)[:, :n_bins]
        score     = torch.fft.irfft(score_f,     n=n_fft, dim=-1)[:, :n_bins]

        # Cubic B-spline weights at each pair's fractional bin offset.
        q_abs_d = q_abs.detach()
        frac    = (q_abs_d - t0_ref) / tick_ns
        bin_lo  = frac.floor().long()                       # (M, Q)
        f       = frac - bin_lo                              # (M, Q) ∈ [0, 1)
        omf     = 1.0 - f
        inv6    = 1.0 / 6.0
        bw_m1   = inv6 * omf.pow(3)
        bw_0    = inv6 * (4.0 - 6.0 * f.pow(2)   + 3.0 * f.pow(3))
        bw_p1   = inv6 * (4.0 - 6.0 * omf.pow(2) + 3.0 * omf.pow(3))
        bw_p2   = inv6 * f.pow(3)

        M, Q = q_abs.shape
        pmt_expand = pmt_idx.unsqueeze(-1).expand(M, Q)

        def gather_4tap(arr):
            """Gather arr[pmt[m], bin_lo[m,q] + k] · B-spline weight for k=−1..2."""
            res = torch.zeros((M, Q), device=device, dtype=arr.dtype)
            for shift, w in ((-1, bw_m1), (0, bw_0), (1, bw_p1), (2, bw_p2)):
                bin_idx_k = bin_lo + shift
                in_window = (bin_idx_k >= 0) & (bin_idx_k < n_bins)
                bin_idx_kc = bin_idx_k.clamp(0, n_bins - 1)
                vals = arr[pmt_expand, bin_idx_kc] * in_window
                res = res + vals * w
            return res

        grad_hist_at_q = gather_4tap(grad_hist)              # (M, Q)
        score_at_q     = gather_4tap(score)                  # (M, Q)

        grad_q_abs   = -gain * weights.detach() * score_at_q
        grad_weights =  gain * grad_hist_at_q

        # Order matches forward signature: only q_abs and weights receive grad.
        return grad_q_abs, grad_weights, None, None, None, None, None, None, None, None, None


def histogram_and_convolve_pdf(
    sampler,
    pos: torch.Tensor,
    n_photons: torch.Tensor,
    t_step: torch.Tensor,
    *,
    tick_ns: float,
    n_bins: int,
    t0_ref: float,
    kernel_tensor: torch.Tensor,
    gain: float,
    chunk_size: int = 5000,
    expected_eps: float = 1e-9,
    q_stride=None,
):
    """Differentiable forward producing the gain-scaled per-PMT *convolved*
    waveform, with autograd-clean gradients to ``pos``, ``n_photons``, and
    ``t_step`` via ``_HistConvFunction``.

    Equivalent to ``Waveform(adc=histogram_pdf(..)).convolve(kernel, gain)``
    in expectation, but the gradient is C^2 smooth in continuous photon
    arrival time (no kink at integer-tick boundaries).

    Returns a ``(pred, total_pe)`` tuple.  ``pred`` is shape
    ``(n_ch, n_bins + len(kernel) - 1)``, autograd-live.

    Samplers that don't expose ``_emit_chunk`` (e.g. mock samplers used in tests)
    transparently fall back to the legacy ``histogram_pdf`` + ``Waveform.convolve``
    path, which has the same forward behaviour but kink-y per-tick gradients in
    ``t_step``.
    """
    if not hasattr(sampler, "_emit_chunk"):
        # Legacy fallback for non-PCA samplers (no per-source `_emit_chunk`).
        hist = sampler.histogram_pdf(
            pos, n_photons, t_step,
            tick_ns=tick_ns, n_bins=n_bins, t0_ref=t0_ref,
            chunk_size=chunk_size,
        )
        device_h = hist.device
        wf = Waveform(adc=hist, t0=t0_ref, tick_ns=tick_ns,
                      n_channels=hist.shape[0])
        wf = wf.convolve(kernel_tensor, gain)
        total_pe_legacy = hist.sum(dim=1)
        return wf.adc, total_pe_legacy

    device = sampler._device
    n_ch = 2 * sampler._n_pmts
    K = kernel_tensor.shape[0]
    out_len = n_bins + K - 1
    n_fft = _next_fft_size(out_len)

    # Pre-compute kernel_f + kernel_deriv_f once (treated as constants by autograd).
    with torch.no_grad():
        kernel_padded = F.pad(kernel_tensor, (0, n_fft - K))
        kernel_f = torch.fft.rfft(kernel_padded, n=n_fft)
        # Angular frequency vector for d/dt: kernel_deriv_f = i·2π·f · kernel_f.
        freqs = torch.fft.rfftfreq(n_fft, d=tick_ns).to(device=device)
        kernel_deriv_f = (1j * 2.0 * math.pi * freqs) * kernel_f

    # Normalize input shapes (mirrors PCATOFSampler.histogram_pdf preamble).
    pos = pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos)
    pos = pos.to(dtype=torch.float32, device=device)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    N = pos.shape[0]

    if isinstance(n_photons, (int, float)):
        scale = torch.full((N,), float(n_photons) / sampler.n_simulated, device=device)
    else:
        n_ph_t = n_photons if isinstance(n_photons, torch.Tensor) else torch.as_tensor(n_photons)
        scale = n_ph_t.to(dtype=torch.float32, device=device) / sampler.n_simulated

    if t_step is None:
        t_step_t = torch.zeros(N, device=device, dtype=torch.float32)
    else:
        t_step_t = t_step if isinstance(t_step, torch.Tensor) else torch.as_tensor(t_step)
        t_step_t = t_step_t.to(dtype=torch.float32, device=device)

    q_idx, du_eff = sampler._resolve_q_stride(q_stride)

    # Iterate position chunks; collect autograd-live (q_abs, weights, pmt_idx).
    all_q_abs, all_weights, all_pmt = [], [], []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        emit = sampler._emit_chunk(
            pos_chunk=pos[start:end],
            scale_chunk=scale[start:end],
            tns_chunk=t_step_t[start:end],
            du_eff=du_eff, q_idx=q_idx, expected_eps=expected_eps,
        )
        if emit is None:
            continue
        q_abs_c, pmt_c, weights_c = emit
        all_q_abs.append(q_abs_c)
        all_weights.append(weights_c)
        all_pmt.append(pmt_c)

    if not all_q_abs:
        empty_pred = torch.zeros(n_ch, out_len, device=device, dtype=torch.float32)
        empty_pe = torch.zeros(n_ch, device=device, dtype=torch.float32)
        return empty_pred, empty_pe

    q_abs = torch.cat(all_q_abs, dim=0)
    weights = torch.cat(all_weights, dim=0)
    pmt_idx = torch.cat(all_pmt, dim=0)

    pred = _HistConvFunction.apply(
        q_abs, weights, pmt_idx,
        kernel_f, kernel_deriv_f,
        float(t0_ref), float(tick_ns),
        int(n_bins), int(n_ch), float(gain), int(K),
    )

    # total_pe per PMT: sum of in-window weights. Autograd-live through
    # `weights` (so n_photons / pos gradients flow into pe_counts), but the
    # integer in-window mask is detached.
    bin_idx_pe = ((q_abs.detach() - t0_ref) / tick_ns).floor().long()
    in_w_pe = (bin_idx_pe >= 0) & (bin_idx_pe < n_bins)
    per_pair = (weights * in_w_pe).sum(dim=-1)
    total_pe = torch.zeros(n_ch, device=device, dtype=torch.float32)
    total_pe = total_pe.scatter_add(0, pmt_idx, per_pair)
    return pred, total_pe


def time_group_segments(
    t_step: torch.Tensor, gap_threshold_ns: float
) -> List[torch.Tensor]:
    """Split a list of segment `t_step` values into groups localized in time.

    Sorts segments by `t_step` and splits at any gap larger than
    `gap_threshold_ns`. Each returned tensor is a list of indices into the
    original `t_step` array;

    Used by `DifferentiableOpticalSimulator._simulate` to bound autograd
    memory: each time-group is histogrammed + convolved + checkpointed
    independently, so the per-group `(M, Q)` ghost-photon transients never
    accumulate to `(N·P, Q)` across the full event.

    The stochastic `OpticalSimulator` does *not* call this — it samples
    categorically across all segments in one pass and then chunks the
    resulting per-channel *photon arrivals* via `SlicedWaveform.from_photons`'s
    gap-detection. Both simulators produce a `SlicedWaveform`, but cluster
    activity at different stages of the pipeline (pre-emission segments here,
    post-emission per-channel arrivals there).

    Note: `time_groups` (input segments, this helper) and `chunks`
    (entries on the output `SlicedWaveform`) are different concepts —
    typically one time-group produces `n_active_PMTs` chunks.
    """
    if t_step.numel() == 0:
        return []
    order = t_step.detach().argsort()
    t_sorted = t_step[order].detach()
    gaps = torch.diff(t_sorted)
    split_points = (gaps > gap_threshold_ns).nonzero(as_tuple=True)[0] + 1
    return list(torch.tensor_split(order, split_points.cpu()))


def as_dlpack(tensor: ArrayLike) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor
    return torch.from_dlpack(tensor)

class DifferentiableOpticalSimulator(OpticalSimulator):
    """OpticalSimulator with stochastic delays replaced by kernel convolution.

    The delay PDFs (Scintillation, TPB, TTS) must already be folded into
    `config.kernel` — typically a `Response` composite.  Instead of
    drawing per-photon delays, the photon histogram is convolved with the
    full delay-and-detector response in a single FFT pass.

    All other stochastic operations are not allowed, with the exception of
    the digitization step.
    """

    def __init__(self, config: OpticalSimConfig):
        if not hasattr(config.tof_sampler, "sample_pdf"):
            raise ValueError(
                "DifferentiableOpticalSimulator requires a TOF sampler that exposes "
                "`sample_pdf(...)` (e.g. PCATOFSampler subclasses). Got "
                f"{type(config.tof_sampler).__name__}."
            )
        super().__init__(config)

    def simulate(
        self,
        pos: torch.Tensor,
        n_photons: torch.Tensor,
        t_step: torch.Tensor,
        stitched: bool = True,
        subtract_t0: bool = False,
        add_baseline_noise: bool = True,
    ) -> Union[SlicedWaveform, Waveform]:
        """Group segments by time proximity, build a
        small dense histogram per group via `histogram_pdf` (checkpoint-ed),
        convolve each with a small FFT, and assemble into a `SlicedWaveform`.

        Peak memory is independent of both `N` (segment count, via
        per-chunk checkpointing inside `histogram_pdf`) and total event
        time span (via time-grouping + per-group FFTs instead of one
        monolithic FFT).
        """
        # auto-convert any array supporting __dlpack__ to torch.Tensor
        pos, n_photons, t_step = map(as_dlpack, (pos, n_photons, t_step))

        if subtract_t0:
            t_step = t_step - t_step.min()

        cfg = self.config
        device = self._device
        fine_tick = self._fine_tick
        fine_kernel_tensor = self._fine_kernel()
        sampler = cfg.tof_sampler
        n_ch = cfg.n_channels

        # Most PCA samplers expose `t_max_ns` as a property; a generic mock
        # sampler may not. Default to 600 ns (the shipped basis's window).
        t_window = getattr(sampler, "t_max_ns", 600.0)
        kernel_extent_ns = float(fine_kernel_tensor.shape[0]) * fine_tick
        gap_threshold = kernel_extent_ns + t_window + _GAP_THRESHOLD_PAD_NS

        # ---- 1. Group segments by t_step proximity ----
        if t_step.numel() == 0:
            return SlicedWaveform(
                adc=torch.zeros(0, device=device),
                offsets=torch.tensor([0], device=device, dtype=torch.long),
                t0_ns=torch.zeros(0, device=device),
                pmt_id=torch.zeros(0, device=device, dtype=torch.long),
                tick_ns=fine_tick, n_channels=n_ch,
                attrs={"pe_counts": torch.zeros(n_ch, device=device)},
            )

        time_groups = time_group_segments(t_step, gap_threshold)

        # Pre-compute every group's t0_g and n_bins_g in two batched ops, then
        # one combined .tolist() — collapses 2 host-syncs per group (.min().item()
        # and .max().item()) into 2 syncs total.
        group_min = torch.stack([t_step[g].detach().min() for g in time_groups])
        group_max = torch.stack([t_step[g].detach().max() for g in time_groups])
        t0_starts = (group_min / fine_tick).floor() * fine_tick
        n_bins_t = (
            ((group_max - t0_starts + t_window) / fine_tick).floor().long()
            + _N_BINS_GROUP_PAD
        )
        t0_starts_cpu = t0_starts.tolist()
        n_bins_cpu = n_bins_t.tolist()

        # ---- 2. Per-group: histogram_pdf → small Waveform → convolve -----
        all_adc = []
        all_offsets = [0]
        all_t0 = []
        all_pmt = []
        total_pe = torch.zeros(n_ch, device=device, dtype=torch.float32)

        for gi, g_idx in enumerate(time_groups):
            t_g = t_step[g_idx]
            t0_g = float(t0_starts_cpu[gi])
            n_bins_g = int(n_bins_cpu[gi])

            # Aux photons for this group's time window — added to the histogram
            # before convolution. We materialise them as a small dense tensor
            # of shape (n_ch, n_bins_g) so we can fold them into the convolved
            # output by convolving separately and adding (linearity of FFT).
            aux_hist_g = None
            if cfg.aux_photon_sources:
                t_start_g = t0_g
                t_end_g = t0_g + n_bins_g * fine_tick
                for source in cfg.aux_photon_sources:
                    aux_t, aux_ch = source.sample(n_ch, t_start_g, t_end_g, device)
                    if aux_t.numel() == 0:
                        continue
                    aux_bin = ((aux_t - t0_g) / fine_tick).long().clamp(0, n_bins_g - 1)
                    if aux_hist_g is None:
                        aux_hist_g = torch.zeros(
                            n_ch, n_bins_g, device=device, dtype=torch.float32,
                        )
                    aux_flat = aux_ch.long() * n_bins_g + aux_bin
                    aux_hist_g.view(-1).scatter_add_(
                        0, aux_flat, torch.ones_like(aux_t, dtype=aux_hist_g.dtype),
                    )

            # Forward via the option-3 fused custom Function: smooth gradient
            # in t_step (and pos / n_photons) at sub-tick resolution without
            # ever differentiating through the floor cast.
            pred_g, total_pe_g = histogram_and_convolve_pdf(
                sampler,
                pos[g_idx], n_photons[g_idx], t_g,
                tick_ns=fine_tick, n_bins=n_bins_g, t0_ref=t0_g,
                kernel_tensor=fine_kernel_tensor,
                gain=cfg.gain,
                chunk_size=cfg.stream_chunk_size,
            )

            if aux_hist_g is not None:
                aux_wf = Waveform(
                    adc=aux_hist_g, t0=t0_g, tick_ns=fine_tick, n_channels=n_ch,
                ).convolve(fine_kernel_tensor, cfg.gain)
                pred_g = pred_g + aux_wf.adc
                total_pe_g = total_pe_g + aux_hist_g.sum(dim=1)

            total_pe = total_pe + total_pe_g

            wf_g = Waveform(
                adc=pred_g, t0=t0_g, tick_ns=fine_tick, n_channels=n_ch,
            )
            if cfg.oversample > 1:
                wf_g = wf_g.downsample(cfg.oversample)

            # Extract per-PMT chunks, skipping zero channels — vectorized.
            active_mask = wf_g.adc.detach().abs().amax(dim=1) > 1e-12
            active_chs = active_mask.nonzero(as_tuple=True)[0]
            n_active = active_chs.numel()
            if n_active > 0:
                active_adc = wf_g.adc[active_chs]
                chunk_len = active_adc.shape[1]
                all_adc.append(active_adc.reshape(-1))
                base = all_offsets[-1]
                all_offsets.extend(base + (i + 1) * chunk_len for i in range(n_active))
                all_t0.extend([wf_g.t0] * n_active)
                all_pmt.extend(active_chs.tolist())

        # ---- 3. Assemble SlicedWaveform -----------------------------------
        tick_out = fine_tick * cfg.oversample if cfg.oversample > 1 else fine_tick
        sw = SlicedWaveform(
            adc=torch.cat(all_adc) if all_adc else torch.zeros(0, device=device),
            offsets=torch.tensor(all_offsets, device=device, dtype=torch.long),
            t0_ns=torch.tensor(all_t0, device=device, dtype=torch.float32),
            pmt_id=torch.tensor(all_pmt, device=device, dtype=torch.long),
            tick_ns=tick_out,
            n_channels=n_ch,
            attrs={"pe_counts": total_pe},
        )

        if add_baseline_noise and cfg.baseline_noise_std > 0:
            sw.adc = sw.adc + torch.randn_like(sw.adc) * cfg.baseline_noise_std

        if cfg.digitization is not None:
            sw.adc = digitize_ste(
                sw.adc, cfg.digitization.pedestal, cfg.digitization.n_bits,
            )
            sw.attrs["pedestal"] = cfg.digitization.pedestal

        return sw if stitched else sw.deslice()

