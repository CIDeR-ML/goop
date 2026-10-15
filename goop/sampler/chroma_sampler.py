"""TOF sampler that propagates every photon through a Chroma detector.

Instead of looking photons up in a photon library, ``ChromaTOFSampler`` emits
each step's photons into a chroma-lar detector and simulates them with
Chroma's Triton backend (TriChroma): exact detector physics, with no voxel
binning, interpolation or PCA. It needs chroma-lar (which brings TriChroma)
and a CUDA GPU; it is not differentiable.
"""

from __future__ import annotations

import math

import torch

from ..base import TOFSamplerBase


class ChromaTOFSampler(TOFSamplerBase):
    """Sample detected photons by simulating them in a chroma-lar detector.

    Each step emits Poisson(``n_photons * pmt_qe``) isotropic, randomly
    polarised photons at its position and time. Applying the constant PMT
    efficiency before propagation instead of at detection leaves the detected
    photons statistically unchanged and simulates fewer photons. Channel ids
    are the detector's; for chroma-lar's TPC they match the photon library's
    2P layout.
    """

    def __init__(self, detector_config: str = "detector_config_reflect_reflect3wires",
                 wavelength: float = 450.0, pmt_qe: float = 0.12, seed: int | None = None,
                 photons_per_batch: int = 50_000_000, max_steps: int = 1000, device: str = "cuda:0"):
        from chroma.backend import backend_name

        if backend_name() != "triton":
            raise RuntimeError("ChromaTOFSampler needs Chroma's Triton backend; unset CHROMA_BACKEND")
        from chroma.event import Photons
        from chroma.sim import Simulation
        from chroma_lar.geometry import build_detector_from_config

        if not hasattr(Photons(), "index"):
            raise RuntimeError("ChromaTOFSampler needs a TriChroma whose hits carry the input photon index "
                               "(youngsm/trichroma#3)")

        self._device = torch.device(device)
        detector = build_detector_from_config(detector_config, flatten=True)
        self._n_channels = int(detector.num_channels())
        self._sim = Simulation(detector, seed=seed, cuda_device=self._device.index)
        self._wavelength = float(wavelength)
        self._pmt_qe = float(pmt_qe)
        self._photons_per_batch = int(photons_per_batch)
        self._max_steps = int(max_steps)

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @torch.no_grad()
    def sample(self, pos, n_photons, t_step):
        dev = self._device
        pos = torch.as_tensor(pos, dtype=torch.float32, device=dev).reshape(-1, 3)
        n_photons = torch.as_tensor(n_photons, dtype=torch.float32, device=dev).detach()
        n_photons = n_photons.expand(len(pos)) if n_photons.dim() == 0 else n_photons.reshape(-1)
        t_step = (torch.zeros(len(pos), device=dev) if t_step is None
                  else torch.as_tensor(t_step, dtype=torch.float32, device=dev).detach().reshape(-1))

        counts = torch.poisson(n_photons * self._pmt_qe).long()
        source = torch.repeat_interleave(torch.arange(len(pos), device=dev), counts)  # step of each photon
        total = len(source)
        if total == 0:
            empty = torch.empty(0, device=dev)
            return empty, empty.long(), empty.long()

        photons = self._photons(pos[source], t_step[source])
        times, channels, rows = [], [], []
        # one event per chunk of at most photons_per_batch photons
        n_chunks = max(1, math.ceil(total / self._photons_per_batch))
        starts = [k * total // n_chunks for k in range(n_chunks + 1)]
        events = self._events(photons, starts)
        for start, ev in zip(starts, self._sim.simulate(events, keep_flat_hits=True, keep_hits=False,
                                                         photons_per_batch=self._photons_per_batch,
                                                         max_steps=self._max_steps)):
            hits = ev.flat_hits
            times.append(torch.as_tensor(hits.t, device=dev))
            channels.append(torch.as_tensor(hits.channel.astype("int64"), device=dev))
            rows.append(torch.as_tensor(hits.index, device=dev) + start)
        return torch.cat(times), torch.cat(channels), source[torch.cat(rows)]

    def _photons(self, pos, t):
        """Isotropic, randomly polarised photons at ``pos`` and times ``t``."""
        from chroma.triton.sources import SourcePhotons

        n, dev = len(pos), self._device
        u = torch.rand((4, n), device=dev)
        cos = u[0:2] * 2 - 1
        sin = (1 - cos * cos).sqrt()
        phi = u[2:4] * (2 * math.pi)
        unit = torch.stack([sin * phi.cos(), sin * phi.sin(), cos], 2)
        direction = unit[0]
        pol = torch.linalg.cross(direction, unit[1])
        pol /= pol.norm(dim=1, keepdim=True)
        return SourcePhotons(
            pos=pos, dir=direction, pol=pol, wavelengths=torch.full((n,), self._wavelength, device=dev), t=t,
            last_hit_triangles=torch.full((n,), -1, device=dev, dtype=torch.int32),
            flags=torch.zeros(n, device=dev, dtype=torch.int32), weights=torch.ones(n, device=dev),
            evidx=torch.zeros(n, device=dev, dtype=torch.int32))

    @staticmethod
    def _events(photons, starts):
        from chroma.event import Event

        for start, end in zip(starts[:-1], starts[1:]):
            yield Event(photons_beg=photons[start:end])


def create_chroma_tof_sampler(**kwargs) -> ChromaTOFSampler:
    """Create a ChromaTOFSampler (chroma-lar's wire TPC at 450 nm, PMT efficiency 0.12)."""
    return ChromaTOFSampler(**kwargs)
