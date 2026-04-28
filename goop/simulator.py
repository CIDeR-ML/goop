"""Optical TPC simulation pipeline (stochastic + expectation, in one class).

A single :class:`OpticalSimulator` drives both modes, picked at call time
via :meth:`OpticalSimulator.simulate`:

- ``stochastic=True`` (default): Poisson photon sampling + per-photon delays.
- ``stochastic=False``: deterministic PDF deposition; differentiable wrt
  ``pos`` and ``n_photons``. Configure ``cfg.kernel`` to fold in delay PDFs
  (typically ``create_default_response()``); ``cfg.tof_sampler`` must be a
  ``PCATOFSampler`` subclass (any current TOF sampler qualifies).

Output is a ``SlicedWaveform`` (default ``sliced=True``) or a dense
``Waveform`` (``sliced=False``). Optional per-segment ``labels`` returns a
list of waveforms, one per label — supported in both modes.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
import warnings

import torch

from .base import ConvolutionKernelBase, DelaySamplerBase, PhotonSourceBase, TOFSamplerBase
from .digitize import DigitizationConfig, digitize
from .sampler import create_default_tof_sampler
from .delays import Delays, create_default_delays
from .kernels import create_default_kernel
from .waveform import SlicedWaveform, Waveform

# Any array supporting __dlpack__ (torch.Tensor, jax.Array, cupy.ndarray, ...).
ArrayLike = Union[torch.Tensor, Any]

# Padding for streaming-PDF time-island grouping. Gap threshold pads beyond
# kernel_extent + sampler t-window so a split can never bisect a
# kernel-extent tail. Per-island n_bins padding absorbs integer rounding
# around the late-photon tail.
_GAP_THRESHOLD_PAD_NS = 100.0
_N_BINS_ISLAND_PAD = 10


@dataclass
class OpticalSimConfig:
    """Full configuration for the optical simulation pipeline.

    Most fields have sensible defaults; the common overrides are
    ``tof_sampler`` (provide a SIREN/LUT sampler trained on your detector)
    and ``delays`` (scintillation/TPB/TTS PDFs for stochastic mode, or fold
    them into ``kernel`` as a ``Response`` for expectation mode).
    """

    # Photon transport / response (shared across both modes)
    tof_sampler: TOFSamplerBase = field(default_factory=create_default_tof_sampler)
    delays: Union[Delays, List[DelaySamplerBase]] = field(default_factory=list)
    kernel: ConvolutionKernelBase = field(default_factory=create_default_kernel)
    aux_photon_sources: List[PhotonSourceBase] = field(default_factory=list)
    digitization: Optional[DigitizationConfig] = None

    # Output / device
    device: str = "cuda"
    tick_ns: float = 1.0
    oversample: int = 1
    gain: float = -45.0    # per-PMT gain in ADC units

    # Noise
    ser_jitter_std: float = 0.0      # multiplicative N(1, σ) on per-photon weights
    baseline_noise_std: float = 0.0  # additive N(0, σ) per ADC bin

    # Multi-module geometry: total channels = tof_sampler.n_channels * n_modules
    n_modules: int = 1

    # Labeled-mode policy
    n_labels_to_simulate: int = 3
    time_window_ns: Optional[float] = None

    # Expectation-mode (stochastic=False) tuning
    streaming: bool = True            # use time-island streaming when sliced=True
    pos_batch_size: int = 5000        # positions per gradient checkpoint
    checkpoint: bool = True           # enable per-chunk torch.utils.checkpoint

    def __post_init__(self):
        if not isinstance(self.oversample, int) or self.oversample < 1:
            raise ValueError(f"oversample must be an int >= 1, got {self.oversample}")
        self.n_channels = self.tof_sampler.n_channels * self.n_modules
        if isinstance(self.delays, list):
            self.delays = Delays(self.delays)
        self.n_labels_to_simulate = min(self.n_labels_to_simulate, 30)


class OpticalSimulator:
    """Unified optical TPC simulation pipeline.

    Pick the mode at call time via :meth:`simulate`. Defaults match the
    common case (Poisson stochastic + sliced output, no labels), so most
    callers don't need any kwargs beyond ``pos, n_photons, t_step``.

    Modes (the ``stochastic`` × ``sliced`` matrix):

    - ``stochastic=True, sliced=True``: Poisson + delays → ``SlicedWaveform``.
    - ``stochastic=True, sliced=False``: Poisson + delays → dense ``Waveform``.
    - ``stochastic=False, sliced=True``: PDF deposition (per-time-island
      streaming when ``cfg.streaming``) → ``SlicedWaveform``. Differentiable.
    - ``stochastic=False, sliced=False``: PDF deposition → dense
      ``Waveform``. Differentiable but memory cost is O(n_channels · t_span).

    Pass ``labels=...`` in any cell of the matrix to get a list of
    per-label waveforms.
    """

    def __init__(self, config: OpticalSimConfig):
        self.config = config
        self._device = torch.device(config.device)
        self._fine_tick = config.tick_ns / config.oversample
        if config.oversample > 1:
            self._fine_kernel = config.kernel.with_tick_ns(self._fine_tick)
        else:
            self._fine_kernel = config.kernel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        pos: ArrayLike,
        n_photons: ArrayLike,
        t_step: ArrayLike,
        labels: Optional[ArrayLike] = None,
        *,
        sliced: bool = True,
        stochastic: bool = True,
        subtract_t0: bool = False,
        add_baseline_noise: bool = True,
        label_batch_size: Optional[int] = None,
        return_tpc: bool = False,
    ) -> Union[SlicedWaveform, Waveform, List[SlicedWaveform]]:
        """Run the simulation pipeline.

        Parameters
        ----------
        pos, n_photons, t_step : array-like
            Per-segment ``(N, 3)`` positions in mm, photon yields ``(N,)``,
            and emission times ``(N,)`` in ns. Any ``__dlpack__``-supporting
            array works (torch / JAX / CuPy).
        labels : optional ``(N,)`` long
            Per-segment label (e.g. volume id, interaction id). When given,
            returns ``list[SlicedWaveform]`` (one per label). Negative
            label values are dropped.
        sliced : bool, default ``True``
            ``True``: ``SlicedWaveform`` output (CSR; memory-efficient for
            sparse events). ``False``: dense ``Waveform`` on a shared time
            axis. Forced to ``True`` when labels are given.
        stochastic : bool, default ``True``
            ``True``: Poisson + per-photon delay sampling, no gradients.
            ``False``: PDF deposition, differentiable wrt ``pos`` and
            ``n_photons``.
        subtract_t0 : bool, default ``False``
            Shift ``t_step`` so its minimum is 0 (per-label minimum if
            labels are given).
        add_baseline_noise : bool, default ``True``
            Apply ``cfg.baseline_noise_std`` Gaussian noise to ADC samples.
        label_batch_size : optional int
            Cap unique labels per virtual-channel batch to limit memory.
            Default ``None`` = process all labels in one batch.
        return_tpc : bool, default ``False``
            When ``True`` with labels, also return the (filtered) input
            arrays for the I/O layer to save alongside the waveforms.

        Returns
        -------
        ``SlicedWaveform`` or ``Waveform`` (no labels), or
        ``list[SlicedWaveform]`` (with labels), or
        ``(list[SlicedWaveform], pos, n_photons, t_step, labels)`` when
        ``return_tpc=True``.
        """
        cfg = self.config
        device = self._device

        if labels is not None and not sliced:
            raise ValueError(
                "simulate(labels=..., sliced=False) is not supported — labels "
                "always produce per-label SlicedWaveforms."
            )

        # Auto-convert any __dlpack__ array to torch.Tensor on device.
        pos, n_photons, t_step = (
            torch.from_dlpack(a) if not isinstance(a, torch.Tensor) else a
            for a in (pos, n_photons, t_step)
        )
        pos = pos.to(device=device, dtype=torch.float32)
        n_photons = n_photons.to(device=device)
        t_step = t_step.to(device=device, dtype=torch.float32)

        unique_tpc_labels: Optional[torch.Tensor] = None
        if labels is not None:
            labels = (
                torch.from_dlpack(labels) if not isinstance(labels, torch.Tensor) else labels
            ).to(dtype=torch.long, device=device)
            unique_tpc_labels = torch.unique(labels[labels >= 0])

        # Per-label or global t0 subtraction.
        if subtract_t0:
            if labels is not None and unique_tpc_labels is not None and unique_tpc_labels.numel() > 0:
                valid_mask = labels >= 0
                label_indices = torch.searchsorted(unique_tpc_labels, labels.clamp(min=0))
                per_label_min = torch.full(
                    (len(unique_tpc_labels),), float("inf"),
                    dtype=t_step.dtype, device=device,
                )
                per_label_min.scatter_reduce_(
                    0, label_indices[valid_mask], t_step[valid_mask],
                    reduce="amin", include_self=True,
                )
                t_step = t_step - per_label_min[label_indices]
            else:
                t_step = t_step - t_step.min()

        # Per-label random t0 within [0, time_window_ns) and time-window filter.
        pos_masked = n_photons_masked = t_step_masked = labels_masked = None
        if labels is not None and cfg.time_window_ns is not None and unique_tpc_labels is not None:
            valid_mask = labels >= 0
            label_indices = torch.searchsorted(unique_tpc_labels, labels[valid_mask])
            rand_t0_per_label = torch.rand(len(unique_tpc_labels), device=device) * cfg.time_window_ns
            t_step = t_step.clone()
            t_step[valid_mask] = t_step[valid_mask] + rand_t0_per_label[label_indices]
            keep = valid_mask & (t_step <= cfg.time_window_ns)
            pos_masked = pos[keep]
            n_photons_masked = n_photons[keep]
            t_step_masked = t_step[keep]
            labels_masked = labels[keep]
            pos, n_photons, t_step, labels = pos_masked, n_photons_masked, t_step_masked, labels_masked

        # Dispatch on (stochastic, sliced + cfg.streaming).
        ctx = torch.no_grad() if stochastic else nullcontext()
        with ctx:
            if stochastic:
                result = self._simulate_stoch(
                    pos, n_photons, t_step, labels=labels, sliced=sliced,
                    add_baseline_noise=add_baseline_noise,
                    label_batch_size=label_batch_size,
                )
            else:
                if sliced and cfg.streaming:
                    result = self._simulate_pdf_streaming(
                        pos, n_photons, t_step, labels=labels,
                        add_baseline_noise=add_baseline_noise,
                        label_batch_size=label_batch_size,
                    )
                else:
                    if not sliced and pos.shape[0] > 10000:
                        warnings.warn(
                            "stochastic=False with sliced=False can OOM for large N "
                            f"(got {pos.shape[0]} segments). Prefer sliced=True for "
                            "the time-island streaming path."
                        )
                    result = self._simulate_pdf_dense(
                        pos, n_photons, t_step, labels=labels, sliced=sliced,
                        add_baseline_noise=add_baseline_noise,
                        label_batch_size=label_batch_size,
                    )

        if return_tpc:
            if labels is None:
                raise ValueError("return_tpc=True requires labels.")
            return (
                result,
                (pos_masked if pos_masked is not None else pos).cpu().numpy(),
                (n_photons_masked if n_photons_masked is not None else n_photons).cpu().numpy(),
                (t_step_masked if t_step_masked is not None else t_step).cpu().numpy(),
                (labels_masked if labels_masked is not None else labels).cpu().numpy(),
            )
        return result

    # ------------------------------------------------------------------
    # Stochastic path: Poisson sampling + per-photon delays
    # ------------------------------------------------------------------

    def _simulate_stoch(
        self, pos, n_photons, t_step, *,
        labels: Optional[torch.Tensor],
        sliced: bool,
        add_baseline_noise: bool,
        label_batch_size: Optional[int],
    ):
        cfg = self.config
        device = self._device

        # 1. TOF sampling (one batched Poisson + inverse-CDF pass).
        times, channels, _, source_idx = cfg.tof_sampler.sample_photons(
            pos, n_photons, t_step=t_step,
            stochastic=True, return_source_idx=True,
        )

        # 2. Stochastic delays.
        if times.numel() > 0:
            times = times + cfg.delays.sample(times.shape[0], device)

        if labels is not None:
            return self._dispatch_labeled(
                times, channels, weights=None, source_idx=source_idx,
                labels=labels, sliced=sliced,
                add_baseline_noise=add_baseline_noise,
                label_batch_size=label_batch_size,
            )

        # Unlabeled: aux photon sources injected globally.
        if cfg.aux_photon_sources and times.numel() > 0:
            t_start, t_end = times.min().item(), times.max().item()
            for source in cfg.aux_photon_sources:
                aux_t, aux_ch = source.sample(cfg.n_channels, t_start, t_end, device)
                if aux_t.numel() > 0:
                    times = torch.cat([times, aux_t])
                    channels = torch.cat([channels, aux_ch])

        return self._simulate_core(
            times, channels, sliced=sliced, weights=None,
            add_baseline_noise=add_baseline_noise, stochastic=True,
        )

    # ------------------------------------------------------------------
    # PDF (expectation) path — non-streaming (dense ghost photons)
    # ------------------------------------------------------------------

    def _simulate_pdf_dense(
        self, pos, n_photons, t_step, *,
        labels: Optional[torch.Tensor],
        sliced: bool,
        add_baseline_noise: bool,
        label_batch_size: Optional[int],
    ):
        cfg = self.config
        device = self._device

        times, channels, weights, source_idx = cfg.tof_sampler.sample_photons(
            pos, n_photons, t_step=t_step,
            stochastic=False, return_source_idx=True,
        )

        if labels is not None:
            return self._dispatch_labeled(
                times, channels, weights=weights, source_idx=source_idx,
                labels=labels, sliced=sliced,
                add_baseline_noise=add_baseline_noise,
                label_batch_size=label_batch_size,
            )

        if cfg.aux_photon_sources and times.numel() > 0:
            t_start, t_end = times.min().item(), times.max().item()
            for source in cfg.aux_photon_sources:
                aux_t, aux_ch = source.sample(cfg.n_channels, t_start, t_end, device)
                if aux_t.numel() > 0:
                    times = torch.cat([times, aux_t])
                    channels = torch.cat([channels, aux_ch])
                    weights = torch.cat([weights, torch.ones_like(aux_t)])

        return self._simulate_core(
            times, channels, sliced=sliced, weights=weights,
            add_baseline_noise=add_baseline_noise, stochastic=False,
        )

    # ------------------------------------------------------------------
    # Labeled-mode dispatcher (shared by stoch + pdf-dense paths)
    # ------------------------------------------------------------------

    def _dispatch_labeled(
        self, times, channels, *,
        weights: Optional[torch.Tensor],
        source_idx: torch.Tensor,
        labels: torch.Tensor,
        sliced: bool,
        add_baseline_noise: bool,
        label_batch_size: Optional[int],
    ) -> List[SlicedWaveform]:
        cfg = self.config
        device = self._device

        photon_labels = (
            labels[source_idx] if times.numel() > 0
            else torch.empty(0, dtype=torch.long, device=device)
        )
        unique_labels = torch.unique(photon_labels[photon_labels >= 0])
        unique_labels = unique_labels[: cfg.n_labels_to_simulate]
        n_labels = len(unique_labels)
        bs = label_batch_size or n_labels or 1

        results: List[SlicedWaveform] = []
        for start in range(0, n_labels, bs):
            batch_labels = unique_labels[start:start + bs]
            results.extend(self._simulate_labeled_batch(
                times, channels, photon_labels, batch_labels,
                weights=weights, sliced=sliced,
                add_baseline_noise=add_baseline_noise,
            ))
        return results

    def _simulate_labeled_batch(
        self, times, channels, photon_labels, batch_labels,
        *,
        weights: Optional[torch.Tensor],
        sliced: bool,
        add_baseline_noise: bool,
    ) -> List[SlicedWaveform]:
        """Run virtual-channel pipeline for a batch of labels."""
        cfg = self.config
        device = self._device
        n_ch = cfg.n_channels
        n_batch = len(batch_labels)

        batch_mask = torch.isin(photon_labels, batch_labels)
        b_times = times[batch_mask]
        b_channels = channels[batch_mask]
        b_photon_labels = photon_labels[batch_mask]
        b_weights = weights[batch_mask] if weights is not None else None

        label_idx = torch.searchsorted(batch_labels, b_photon_labels)
        virtual_channels = label_idx * n_ch + b_channels

        if cfg.aux_photon_sources and b_times.numel() > 0:
            for li in range(n_batch):
                lbl_mask = b_photon_labels == batch_labels[li]
                lbl_t = b_times[lbl_mask]
                if lbl_t.numel() == 0:
                    continue
                t_start, t_end = lbl_t.min().item(), lbl_t.max().item()
                for source in cfg.aux_photon_sources:
                    aux_t, aux_ch = source.sample(n_ch, t_start, t_end, device)
                    if aux_t.numel() > 0:
                        b_times = torch.cat([b_times, aux_t])
                        virtual_channels = torch.cat([virtual_channels, li * n_ch + aux_ch])
                        if b_weights is not None:
                            b_weights = torch.cat([b_weights, torch.ones_like(aux_t)])

        combined = self._simulate_core(
            b_times, virtual_channels, sliced=sliced,
            weights=b_weights, add_baseline_noise=add_baseline_noise,
            stochastic=(weights is None),
            n_channels=n_ch * n_batch,
        )
        return self._split_by_label(combined, n_ch, batch_labels)

    @staticmethod
    def _split_by_label(
        wf: SlicedWaveform, n_channels: int, unique_labels: torch.Tensor,
    ) -> List[SlicedWaveform]:
        """Slice a virtual-channel SlicedWaveform into per-label waveforms."""
        device = wf.adc.device
        results: List[SlicedWaveform] = []
        for li, lbl in enumerate(unique_labels):
            ch_lo = li * n_channels
            ch_hi = ch_lo + n_channels
            idx = ((wf.pmt_id >= ch_lo) & (wf.pmt_id < ch_hi)).nonzero(as_tuple=True)[0]

            if idx.numel() == 0:
                sub = SlicedWaveform(
                    adc=torch.empty(0, device=device),
                    offsets=torch.tensor([0], device=device, dtype=torch.long),
                    t0_ns=torch.empty(0, device=device),
                    pmt_id=torch.empty(0, device=device, dtype=torch.long),
                    tick_ns=wf.tick_ns, n_channels=n_channels,
                    attrs={"pe_counts": wf.attrs["pe_counts"][ch_lo:ch_hi],
                           "label": lbl.item()},
                )
            else:
                starts = wf.offsets[idx]
                ends = wf.offsets[idx + 1]
                lengths = ends - starts
                sample_indices = torch.cat([
                    torch.arange(s, e, device=device)
                    for s, e in zip(starts.tolist(), ends.tolist())
                ])
                sub = SlicedWaveform(
                    adc=wf.adc[sample_indices],
                    offsets=torch.cat([torch.tensor([0], device=device), lengths.cumsum(0)]),
                    t0_ns=wf.t0_ns[idx],
                    pmt_id=wf.pmt_id[idx] - ch_lo,
                    tick_ns=wf.tick_ns, n_channels=n_channels,
                    attrs={"pe_counts": wf.attrs["pe_counts"][ch_lo:ch_hi],
                           "label": lbl.item()},
                )
            results.append(sub)
        return results

    # ------------------------------------------------------------------
    # Core: histogram → SER → convolve → downsample → noise → digitize
    # ------------------------------------------------------------------

    def _simulate_core(
        self, times, channels, *,
        sliced: bool,
        weights: Optional[torch.Tensor],
        add_baseline_noise: bool,
        stochastic: bool,
        n_channels: Optional[int] = None,
    ) -> Union[SlicedWaveform, Waveform]:
        """Histogram (with optional weights and SER jitter) → convolve →
        downsample → optional baseline noise → optional digitize.

        ``weights=None`` → integer ``pe_counts`` via ``bincount``;
        ``weights`` given → float32 ``pe_counts`` via ``scatter_add``.

        Aux photon sources must be injected into ``times``/``channels``
        (and ``weights`` for the PDF path) by the caller.
        """
        cfg = self.config
        device = self._device
        fine_tick = self._fine_tick
        fine_kernel_tensor = self._fine_kernel()
        if n_channels is None:
            n_channels = cfg.n_channels

        # SER jitter: multiplicative N(1, σ) on per-photon weights.
        if cfg.ser_jitter_std > 0 and times.numel() > 0:
            jitter = torch.normal(
                1.0, cfg.ser_jitter_std, (times.shape[0],), device=device,
            )
            weights = jitter if weights is None else weights * jitter

        wvfm_cls = SlicedWaveform if sliced else Waveform
        extra_args: dict = {}
        if sliced:
            extra_args["kernel_extent_ns"] = fine_kernel_tensor.shape[0] * fine_tick
        if cfg.oversample > 1:
            extra_args["t0_snap_ns"] = cfg.tick_ns
        if weights is not None:
            extra_args["weights"] = weights

        if times.numel() > 0:
            if weights is None:
                pe_counts = torch.bincount(channels.long(), minlength=n_channels)
            else:
                pe_counts = torch.zeros(n_channels, device=device, dtype=torch.float32)
                pe_counts = pe_counts.scatter_add(0, channels.long(), weights)
        else:
            dtype = torch.long if weights is None else torch.float32
            pe_counts = torch.zeros(n_channels, device=device, dtype=dtype)

        wf = wvfm_cls.from_photons(times, channels, fine_tick, n_channels, **extra_args)
        wf.attrs["pe_counts"] = pe_counts

        wf = wf.convolve(fine_kernel_tensor, cfg.gain)
        if cfg.oversample > 1:
            wf = wf.downsample(cfg.oversample)

        if add_baseline_noise and cfg.baseline_noise_std > 0:
            # Out-of-place add preserves gradients in PDF mode.
            wf.adc = wf.adc + torch.randn_like(wf.adc) * cfg.baseline_noise_std

        if cfg.digitization is not None:
            wf.adc = digitize(
                wf.adc, cfg.digitization.pedestal, cfg.digitization.n_bits,
                ste=not stochastic,
            )
            wf.attrs["pedestal"] = cfg.digitization.pedestal

        return wf

    # ------------------------------------------------------------------
    # PDF (expectation) path — sliced + streaming (time-island grouping)
    # ------------------------------------------------------------------

    def _simulate_pdf_streaming(
        self, pos, n_photons, t_step, *,
        labels: Optional[torch.Tensor],
        add_baseline_noise: bool,
        label_batch_size: Optional[int],
    ):
        """Time-island streaming PDF deposition. Memory-bounded by per-island
        size, regardless of total event time span. Differentiable.

        With labels, runs once per label-batch with virtual-channel
        ``label_offsets`` so each batch produces one
        ``(n_batch * n_ch, n_bins_g)`` histogram per island.
        """
        cfg = self.config
        n_ch = cfg.n_channels

        if labels is None:
            return self._streaming_pdf_run(
                pos, n_photons, t_step,
                label_offsets=None, n_label_groups=1,
                add_baseline_noise=add_baseline_noise,
            )
        # stream per-label pdfs
        unique_labels = torch.unique(labels[labels >= 0])[: cfg.n_labels_to_simulate]
        n_labels_total = len(unique_labels)
        bs = label_batch_size or n_labels_total or 1

        all_results: List[SlicedWaveform] = []
        for start in range(0, n_labels_total, bs):
            batch_labels = unique_labels[start:start + bs]
            batch_mask = torch.isin(labels, batch_labels)
            b_pos = pos[batch_mask]
            b_n = n_photons[batch_mask]
            b_t = t_step[batch_mask]
            b_lbl = labels[batch_mask]
            label_offsets = torch.searchsorted(batch_labels, b_lbl).long()
            combined = self._streaming_pdf_run(
                b_pos, b_n, b_t,
                label_offsets=label_offsets,
                n_label_groups=len(batch_labels),
                add_baseline_noise=add_baseline_noise,
            )
            all_results.extend(self._split_by_label(combined, n_ch, batch_labels))
        return all_results

    def _streaming_pdf_run(
        self, pos, n_photons, t_step, *,
        label_offsets: Optional[torch.Tensor],
        n_label_groups: int,
        add_baseline_noise: bool,
    ) -> SlicedWaveform:
        """One streaming pass; output has ``n_label_groups * n_ch`` virtual channels."""
        cfg = self.config
        device = self._device
        fine_tick = self._fine_tick
        fine_kernel_tensor = self._fine_kernel()
        sampler = cfg.tof_sampler
        n_ch = cfg.n_channels
        virt_ch = n_label_groups * n_ch

        if cfg.ser_jitter_std > 0:
            warnings.warn(
                "ser_jitter_std is ignored in streaming PDF mode (no per-photon "
                "weights); use cfg.streaming=False if you need SER jitter.",
                stacklevel=2,
            )

        # Some samplers expose ``t_max_ns`` as a property; mocks may not.
        t_window = float(getattr(sampler, "t_max_ns", 600.0))
        kernel_extent_ns = float(fine_kernel_tensor.shape[0]) * fine_tick
        gap_threshold = kernel_extent_ns + t_window + _GAP_THRESHOLD_PAD_NS

        if t_step.numel() == 0:
            return SlicedWaveform(
                adc=torch.zeros(0, device=device),
                offsets=torch.tensor([0], device=device, dtype=torch.long),
                t0_ns=torch.zeros(0, device=device),
                pmt_id=torch.zeros(0, device=device, dtype=torch.long),
                tick_ns=fine_tick, n_channels=virt_ch,
                attrs={"pe_counts": torch.zeros(virt_ch, device=device)},
            )

        # 1. Sort segments by t_step and split at natural time gaps.
        order = t_step.detach().argsort()
        t_sorted = t_step[order].detach()
        gaps = torch.diff(t_sorted)
        split_points = (gaps > gap_threshold).nonzero(as_tuple=True)[0] + 1
        islands = torch.tensor_split(order, split_points.cpu())

        # Pre-compute per-island t0 and n_bins in two batched ops.
        island_min = torch.stack([t_step[g].detach().min() for g in islands])
        island_max = torch.stack([t_step[g].detach().max() for g in islands])
        t0_starts = (island_min / fine_tick).floor() * fine_tick
        n_bins_t = (
            ((island_max - t0_starts + t_window) / fine_tick).floor().long()
            + _N_BINS_ISLAND_PAD
        )
        t0_starts_cpu = t0_starts.tolist()
        n_bins_cpu = n_bins_t.tolist()

        all_adc: List[torch.Tensor] = []
        all_offsets: List[int] = [0]
        all_t0: List[float] = []
        all_pmt: List[int] = []
        total_pe = torch.zeros(virt_ch, device=device, dtype=torch.float32)

        # 2. Per-island: sample_histogram → small Waveform → convolve.
        for gi, g_idx in enumerate(islands):
            t_g = t_step[g_idx]
            t0_g = float(t0_starts_cpu[gi])
            n_bins_g = int(n_bins_cpu[gi])
            offsets_g = label_offsets[g_idx] if label_offsets is not None else None

            hist_g = sampler.sample_histogram(
                pos[g_idx], n_photons[g_idx], t_g,
                stochastic=False,
                tick_ns=fine_tick, n_bins=n_bins_g, t0_ref=t0_g,
                chunk_size=cfg.pos_batch_size,
                use_checkpoint=cfg.checkpoint,
                label_offsets=offsets_g,
            )
            # Pad hist to virt_ch rows when the island only saw a subset of labels.
            if hist_g.shape[0] < virt_ch:
                pad = torch.zeros(
                    virt_ch - hist_g.shape[0], n_bins_g,
                    device=device, dtype=hist_g.dtype,
                )
                hist_g = torch.cat([hist_g, pad], dim=0)
            total_pe = total_pe + hist_g.sum(dim=1)

            # Aux sources: replicate into every label group's channel range.
            if cfg.aux_photon_sources:
                t_start_g = t0_g
                t_end_g = t0_g + n_bins_g * fine_tick
                for source in cfg.aux_photon_sources:
                    aux_t, aux_ch = source.sample(n_ch, t_start_g, t_end_g, device)
                    if aux_t.numel() == 0:
                        continue
                    aux_bin = ((aux_t - t0_g) / fine_tick).long().clamp(0, n_bins_g - 1)
                    for li in range(n_label_groups):
                        aux_flat = (aux_ch.long() + li * n_ch) * n_bins_g + aux_bin
                        hist_g = hist_g.reshape(-1).scatter_add(
                            0, aux_flat, torch.ones_like(aux_t, dtype=hist_g.dtype),
                        ).reshape(virt_ch, n_bins_g)

            wf_g = Waveform(
                adc=hist_g, t0=t0_g, tick_ns=fine_tick, n_channels=virt_ch,
            )
            wf_g = wf_g.convolve(fine_kernel_tensor, cfg.gain)
            if cfg.oversample > 1:
                wf_g = wf_g.downsample(cfg.oversample)

            # Extract per-PMT chunks, skipping zero channels.
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

        # 3. Assemble SlicedWaveform.
        tick_out = fine_tick * cfg.oversample if cfg.oversample > 1 else fine_tick
        sw = SlicedWaveform(
            adc=torch.cat(all_adc) if all_adc else torch.zeros(0, device=device),
            offsets=torch.tensor(all_offsets, device=device, dtype=torch.long),
            t0_ns=torch.tensor(all_t0, device=device, dtype=torch.float32),
            pmt_id=torch.tensor(all_pmt, device=device, dtype=torch.long),
            tick_ns=tick_out,
            n_channels=virt_ch,
            attrs={"pe_counts": total_pe},
        )

        if add_baseline_noise and cfg.baseline_noise_std > 0:
            sw.adc = sw.adc + torch.randn_like(sw.adc) * cfg.baseline_noise_std

        if cfg.digitization is not None:
            sw.adc = digitize(
                sw.adc, cfg.digitization.pedestal, cfg.digitization.n_bits,
                ste=True,
            )
            sw.attrs["pedestal"] = cfg.digitization.pedestal

        return sw
