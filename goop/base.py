"""Abstract base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class TOFSamplerBase(ABC):
    """Base class for time-of-flight samplers.

    Implementations must provide both photon-list and dense-histogram outputs,
    each in two flavours selected by ``stochastic``: Poisson sampling for stochastic mode,
    deterministic PDF deposition for expectation mode.
    """

    @abstractmethod
    def sample_photons(
        self,
        pos: torch.Tensor,
        n_photons: torch.Tensor,
        t_step: torch.Tensor,
        *,
        stochastic: bool = True,
        return_source_idx: bool = False,
    ):
        """Return ``(times, channels, weights[, source_idx])``.

        Stochastic mode: weights are unit (1.0); ``source_idx`` (when
        requested) is the per-photon global position index.
        Expectation mode: weights carry probability mass; the sampler emits
        ``Q`` ghost photons per active (position, PMT) pair.
        """
        ...

    @abstractmethod
    def sample_histogram(
        self,
        pos: torch.Tensor,
        n_photons: torch.Tensor,
        t_step: torch.Tensor,
        *,
        stochastic: bool = True,
        tick_ns: float,
        n_bins: int,
        t0_ref: float = 0.0,
    ) -> torch.Tensor:
        """Return a dense ``(2P, n_bins)`` histogram. Returns ``int32`` for
        ``stochastic=True``, ``float32`` (and differentiable) otherwise."""
        ...

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of output channels."""
        ...

class DelaySamplerBase(ABC):
    """Base class for stochastic time-offset samplers."""

    @abstractmethod
    def __call__(self, n_photons: int, device: torch.device) -> torch.Tensor:
        """Return (n_photons,) tensor of time offsets in ns."""
        ...


class PhotonSourceBase(ABC):
    """Base class for auxiliary photon sources (dark noise, afterpulsing, etc.).

    Unlike DelaySamplerBase (which adds time offsets to existing photons),
    PhotonSourceBase creates NEW photons with their own times and channels.
    Called between delay sampling and histogramming in the pipeline.
    """

    @abstractmethod
    def sample(
        self,
        n_channels: int,
        t_start_ns: float,
        t_end_ns: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate auxiliary photon hits.

        Parameters
        ----------
        n_channels : Total number of PMT channels.
        t_start_ns : Start of the time window in ns.
        t_end_ns : End of the time window in ns.
        device : Torch device.

        Returns
        -------
        times : (M,) photon arrival times in ns.
        channels : (M,) PMT channel IDs.
        """
        ...


class ConvolutionKernelBase(ABC):
    """Base class for impulse-response kernels.

    Subclasses must be @dataclass with tick_ns, device, and _kernel_cache
    attributes, and implement __call__ (returns kernel tensor).
    """

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        """Return 1-D kernel tensor (time domain)."""
        ...

    def with_tick_ns(self, tick_ns: float) -> ConvolutionKernelBase:
        """Return a copy with different tick_ns (cache cleared)."""
        from dataclasses import replace
        new = replace(self, tick_ns=tick_ns)
        new._kernel_cache = None
        return new
