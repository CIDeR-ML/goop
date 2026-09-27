"""Photon time-of-flight samplers.

Class hierarchy:

    TOFSamplerBase (ABC)                          — goop/base.py
     └── PCATOFSampler (ABC)                      — shared PCA reconstruction +
          │                                         Poisson/inverse-CDF sampling +
          │                                         differentiable ``sample_pdf``
          │                                         Abstract: ``_lookup(pos)``
          ├── TOFSampler                          — ``_lookup`` via voxel LUT
          │    │                                    trilinear (or nearest-neighbor)
          │    │                                    interpolation of a PCA-compressed
          │    │                                    photon library (h5)
          │    └── QuantileTOFSampler             — same, over a QUANTILE-native plib
          │                                         (+ QuantileReconMixin)
          └── SirenTOFSampler                     — ``_lookup`` via a pre-trained
                                                    quantile-head SIREN (q3 model)
                                                    (+ QuantileReconMixin)

QuantileReconMixin replaces the PCA expansion with a plain column select, so the
two quantile-native samplers decode identically to each other and differ from
TOFSampler only by the PCA truncation. Default paths come from goop.sites (NERSC or
s3df, auto-detected or pinned with goop.sites.set_site).

All public names are re-exported here so ``from goop.sampler import X`` and
``from goop import X`` both keep working regardless of which submodule
defines ``X``.
"""

from .base import (
    DEFAULT_N_SIMULATED,
    DEFAULT_PLIB_PATH,
    PCATOFSampler,
    QuantileReconMixin,
)
from .lut import (
    DifferentiableTOFSampler,
    QuantileTOFSampler,
    TOFSampler,
    create_default_tof_sampler,
    create_quantile_tof_sampler,
)
from .siren import (
    DEFAULT_CFG_PATH,
    DEFAULT_CKPT_PATH,
    DEFAULT_SIRENTV_SRC,
    SirenTOFSampler,
    create_siren_tof_sampler,
)

__all__ = [
    "PCATOFSampler",
    "QuantileReconMixin",
    "TOFSampler",
    "QuantileTOFSampler",
    "DifferentiableTOFSampler",
    "SirenTOFSampler",
    "create_default_tof_sampler",
    "create_quantile_tof_sampler",
    "create_siren_tof_sampler",
    "DEFAULT_PLIB_PATH",
    "DEFAULT_N_SIMULATED",
    "DEFAULT_CKPT_PATH",
    "DEFAULT_CFG_PATH",
    "DEFAULT_SIRENTV_SRC",
]
