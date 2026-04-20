"""Differentiable optical-simulation pipeline.

Phase 1: replaces stochastic per-photon delay sampling with deterministic
convolution against a composite ``Response`` kernel that includes the
delay PDFs.  TOFSampler is still stochastic in this phase (Phase 2 work).

Usage
-----
The simulator reuses ``OpticalSimConfig`` and most of ``OpticalSimulator``
unchanged; the only structural difference is that ``simulate()`` here skips
the ``cfg.delays.sample(...)`` step and is **not** wrapped in
``@torch.no_grad()``.  The kernel must be a ``Response`` (or any
``ConvolutionKernelBase``) that already encodes the delay PDFs the user
wants — typically ``create_default_response()``.

Non-differentiable knobs (``digitization``, ``ser_jitter_std``,
``baseline_noise_std``, ``aux_photon_sources``) must be off; the constructor
raises if any are set.  See ``notes/differentiable.md`` Phase 3 for plans
to relax this.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch

from .simulator import OpticalSimConfig, OpticalSimulator
from .waveform import SlicedWaveform, Waveform

ArrayLike = Union[torch.Tensor, Any]

def as_dlpack(tensor: ArrayLike) -> torch.Tensor:
    """Convert a __dlpack__-supporting array to a torch.Tensor, no-op if already one.

    Note: ``torch.from_dlpack`` fails on tensors that require_grad, so we
    must short-circuit on torch.Tensor inputs to preserve the autograd graph.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor
    return torch.from_dlpack(tensor)

class DifferentiableOpticalSimulator(OpticalSimulator):
    """OpticalSimulator with stochastic delays replaced by kernel convolution.

    The delay PDFs (Scintillation, TPB, TTS) must already be folded into
    ``config.kernel`` — typically a ``Response`` composite.  Instead of
    drawing per-photon delays, the photon histogram is convolved with the
    full delay-and-detector response in a single FFT pass.
    """

    def __init__(self, config: OpticalSimConfig):
        _assert_differentiable(config)
        super().__init__(config)

    def simulate(
        self,
        pos: ArrayLike,
        n_photons: ArrayLike,
        t_step: ArrayLike,
        labels: Optional[ArrayLike] = None,
        stitched: bool = True,
        subtract_t0: bool = False,
        add_baseline_noise: bool = True,
        label_batch_size: Optional[int] = None,
    ) -> Union[SlicedWaveform, Waveform, List[SlicedWaveform]]:
        """Same as ``OpticalSimulator.simulate`` but skips delay sampling and
        does **not** disable autograd.  The delay PDFs are expected to be
        baked into ``cfg.kernel``, as a ``Response``.

        ``stitched`` must be ``True``.

        TOF sampling: if ``cfg.tof_sampler`` exposes a ``sample_pdf(...)``
        method (e.g. ``DifferentiableTOFSampler``), the diff sim routes
        through it for end-to-end differentiability through photon yield
        and voxel interpolation.  Otherwise it falls back to the stochastic
        ``sample()`` and the only differentiable step is the kernel
        convolution (Phase-1 mode).
        """
        if not stitched:
            raise ValueError(
                "DifferentiableOpticalSimulator.simulate(..., stitched=False) is "
                "not supported."
            )
        cfg = self.config
        device = self._device

        # auto-convert any array supporting __dlpack__ to torch.Tensor
        pos, n_photons, t_step = map(as_dlpack, (pos, n_photons, t_step))
        if labels is not None:
            labels = as_dlpack(labels).to(dtype=torch.long, device=device)

        if subtract_t0:
            t_step = t_step - t_step.min()

        use_pdf = hasattr(cfg.tof_sampler, "sample_pdf")

        if use_pdf:
            if labels is not None:
                raise NotImplementedError(
                    "Labeled mode is not yet supported with sample_pdf — "
                    "the PDF-deposition path doesn't track per-photon source "
                    "positions."
                )
            times, channels, weights = cfg.tof_sampler.sample_pdf(
                pos, n_photons, t_step=t_step
            )
        else:
            times, channels, source_idx = cfg.tof_sampler.sample(
                pos, n_photons, t_step=t_step
            )
            weights = None

        # Labeled mode (only via the stochastic sample() fallback)
        if labels is not None:
            photon_labels = (
                labels[source_idx]
                if times.numel() > 0
                else torch.empty(0, dtype=torch.long, device=device)
            )
            unique_labels = torch.unique(labels)
            n_labels = len(unique_labels)
            bs = label_batch_size or n_labels

            results: List[SlicedWaveform] = []
            for start in range(0, n_labels, bs):
                batch_labels = unique_labels[start:start + bs]
                results.extend(
                    self._simulate_labeled_batch(
                        times, channels, photon_labels, batch_labels,
                        stitched, add_baseline_noise,
                    )
                )
            return results

        # Unlabeled mode — aux_photon_sources is asserted empty in __init__,
        # so the injection block from OpticalSimulator.simulate is a no-op
        # here and is omitted entirely.
        return self._simulate(
            times, channels, stitched, add_baseline_noise=add_baseline_noise,
            weights=weights,
        )

    def _simulate(
        self,
        times: torch.Tensor,
        channels: torch.Tensor,
        stitched: bool,
        *,
        n_channels: Optional[int] = None,
        add_baseline_noise: bool = True,
        weights: Optional[torch.Tensor] = None,
    ) -> Union[SlicedWaveform, Waveform]:
        """Histogram (with optional per-photon weights) → convolve → downsample.

        Same shape as ``OpticalSimulator._simulate`` but plumbs ``weights``
        through to ``from_photons`` (needed for the PDF-deposition path,
        where each synthetic "photon" carries its quantile-bin probability
        mass) and uses a weighted scatter for ``pe_counts``.

        Skips the SER-jitter / baseline-noise / digitization branches —
        those are asserted off in the diff sim's ``__init__``.
        """
        cfg = self.config
        device = self._device
        fine_tick = self._fine_tick
        fine_kernel_tensor = self._fine_kernel()
        if n_channels is None:
            n_channels = cfg.n_channels

        wvfm_cls = SlicedWaveform if stitched else Waveform
        extra_args: dict = {}
        if stitched:
            extra_args["kernel_extent_ns"] = fine_kernel_tensor.shape[0] * fine_tick
        if cfg.oversample > 1:
            extra_args["t0_snap_ns"] = cfg.tick_ns
        if weights is not None:
            extra_args["weights"] = weights

        if times.numel() > 0:
            if weights is not None:
                pe_counts = torch.zeros(n_channels, device=device, dtype=torch.float32)
                pe_counts.scatter_add_(0, channels.long(), weights.detach())
            else:
                pe_counts = torch.bincount(channels.long(), minlength=n_channels)
        else:
            pe_counts = torch.zeros(n_channels, device=device, dtype=torch.float32)

        wf = wvfm_cls.from_photons(times, channels, fine_tick, n_channels, **extra_args)
        wf.attrs["pe_counts"] = pe_counts

        wf = wf.convolve(fine_kernel_tensor, cfg.gain)
        if cfg.oversample > 1:
            wf = wf.downsample(cfg.oversample)
        return wf


def _assert_differentiable(cfg: OpticalSimConfig) -> None:
    """Reject configs that contain operations that would block gradients."""
    bad: List[str] = []
    if cfg.digitization is not None:
        bad.append(
            "digitization is not None — round/clamp are non-differentiable"
        )
    if cfg.ser_jitter_std != 0:
        bad.append(
            f"ser_jitter_std={cfg.ser_jitter_std} > 0 — multiplicative Gaussian sampling is stochastic"
        )
    if cfg.baseline_noise_std != 0:
        bad.append(
            f"baseline_noise_std={cfg.baseline_noise_std} > 0 — additive Gaussian sampling is stochastic"
        )
    if cfg.aux_photon_sources:
        bad.append(
            f"aux_photon_sources is non-empty ({len(cfg.aux_photon_sources)} sources) — Poisson sampling is stochastic"
        )
    if bad:
        raise ValueError(
            "DifferentiableOpticalSimulator requires non-differentiable ops "
            "to be off; got:\n  - " + "\n  - ".join(bad)
            + "\nSee notes/differentiable.md Phase 3 for the plan to relax this."
        )
