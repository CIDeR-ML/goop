"""Utility functions for input preprocessing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def voxelize(
    pos: np.ndarray | torch.Tensor,
    n_photons: np.ndarray | torch.Tensor,
    t_step: np.ndarray | torch.Tensor,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin segments into cubic voxels of side length ``dx`` mm.

    Within each voxel, photon counts are summed, and positions / emission
    times are averaged weighted by photon count. Total photon yield is
    exactly preserved.

    Parameters
    ----------
    pos : (N, 3) array — segment positions in mm.
    n_photons : (N,) array — photon count per segment.
    t_step : (N,) array — emission time per segment in ns.
    dx : float — voxel side length in mm. Must be > 0.

    Returns
    -------
    pos_vox : (M, 3) float32 — photon-weighted centroid per voxel.
    nph_vox : (M,) int64 — summed photon count per voxel.
    tns_vox : (M,) float32 — photon-weighted mean emission time per voxel.
    """
    if dx <= 0:
        raise ValueError(f"voxel size dx must be > 0, got {dx}")

    # Convert to numpy if torch tensors
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if isinstance(n_photons, torch.Tensor):
        n_photons = n_photons.detach().cpu().numpy()
    if isinstance(t_step, torch.Tensor):
        t_step = t_step.detach().cpu().numpy()

    pos = np.asarray(pos, dtype=np.float32)
    n_photons = np.asarray(n_photons)
    t_step = np.asarray(t_step, dtype=np.float32)

    # Integer voxel indices
    vox_idx = np.floor(pos / dx).astype(np.int32)

    # Pack (ix, iy, iz) into a single int64 key for grouping.
    # Primes chosen to avoid collisions for detector-scale coordinates.
    keys = (vox_idx[:, 0].astype(np.int64) * 1_000_003
            + vox_idx[:, 1].astype(np.int64) * 1_009
            + vox_idx[:, 2].astype(np.int64))

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_vox = len(unique_keys)

    # Accumulate weighted sums (float64 for precision)
    w = n_photons.astype(np.float64)
    pos_sum = np.zeros((n_vox, 3), dtype=np.float64)
    tns_sum = np.zeros(n_vox, dtype=np.float64)
    w_sum = np.zeros(n_vox, dtype=np.float64)
    nph_sum = np.zeros(n_vox, dtype=np.int64)

    np.add.at(pos_sum, inverse, pos * w[:, None])
    np.add.at(tns_sum, inverse, t_step * w)
    np.add.at(w_sum, inverse, w)
    np.add.at(nph_sum, inverse, n_photons.astype(np.int64))

    mask = w_sum > 0
    pos_vox = (pos_sum[mask] / w_sum[mask, None]).astype(np.float32)
    tns_vox = (tns_sum[mask] / w_sum[mask]).astype(np.float32)
    nph_vox = nph_sum[mask]

    return pos_vox, nph_vox, tns_vox
