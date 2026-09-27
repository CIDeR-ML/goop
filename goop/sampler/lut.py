"""
Voxel-LUT TOF sampler.

Reads a compressed photon library h5 and looks up ``(vis, t0, coeffs)`` per
voxel via trilinear (or nearest-neighbor) interpolation.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from ..base import TOFSamplerBase
from .base import (
    DEFAULT_N_SIMULATED,
    DEFAULT_PLIB_PATH,
    PCATOFSampler,
    QuantileReconMixin,
)

__all__ = [
    "TOFSampler",
    "DifferentiableTOFSampler",
    "QuantileTOFSampler",
    "create_default_tof_sampler",
    "create_quantile_tof_sampler",
]


def create_default_tof_sampler(**kwargs) -> TOFSamplerBase:
    """Create a TOFSampler with the standard photon library.

    Note: ``differentiable`` is accepted for backward compatibility but is a no-op —
    every ``PCATOFSampler`` exposes both ``sample`` and ``sample_pdf``.
    """
    plib_path = kwargs.pop("plib_path", kwargs.pop("filepath", DEFAULT_PLIB_PATH))
    default_kwargs = {
        "n_simulated": DEFAULT_N_SIMULATED,
        "lazy": False,
        "device": "cuda:0",
        "interpolate": True,
        "pmt_qe": 0.12,  # incl. TPB reemission, see https://link.springer.com/article/10.1140/epjc/s10052-024-13306-3
    }
    default_kwargs.update(kwargs)
    default_kwargs.pop("differentiable", None)  # back-compat no-op
    return TOFSampler(plib_path, **default_kwargs)


class TOFSampler(PCATOFSampler):
    """
    Monte Carlo photon time-of-flight sampler from a compressed photon library.

    Reads a half-detector (P PMT) compressed plib and produces full 2P-PMT
    results via x-reflection symmetry. Entirely GPU-native when lazy=False
    and device="cuda".
    """

    def __init__(self, filepath, n_simulated=1.5e7, lazy=True, device="cpu", interpolate=True, pmt_qe=None):
        dev = torch.device(device)
        self._lazy = lazy
        self._interpolate = interpolate
        self._file = None  # set below or kept None for eager mode

        with h5py.File(filepath, "r") as f:
            # Populate shared PCA fields via base helper
            basis = PCATOFSampler._read_h5_basis(filepath)
            # h5py re-open below is avoided by reading LUT tensors in this same block
            self._init_common(
                device=dev,
                n_simulated=n_simulated,
                pmt_qe=float(pmt_qe) if pmt_qe is not None else 1.0,
                n_pmts=basis["n_pmts"],
                n_components=basis["n_components"],
                log_quantile_C=basis["log_quantile_C"],
                t_max_ns=basis["t_max_ns"],
                mode=basis["mode"],
                pca_mean=basis["pca_mean"],
                pca_components=basis["pca_components"],
                u_grid=basis["u_grid"],
                numvox=basis["numvox"],
                min_xyz=basis["min_xyz"],
                max_xyz=basis["max_xyz"],
            )
            self._n_voxels = basis["n_voxels"]

            if not lazy:
                self.vis = torch.from_numpy(f["vis"][:]).float().to(self._device)
                self.t0 = torch.from_numpy(f["t0"][:]).float().to(self._device)
                self.coeffs = torch.from_numpy(f["coeffs"][:]).float().to(self._device)
            else:
                self.vis = self.t0 = self.coeffs = None

        self._file = h5py.File(filepath, "r", swmr=True) if lazy else None

    @classmethod
    def from_arrays(
        cls,
        *,
        vis: torch.Tensor,
        t0: torch.Tensor,
        coeffs: torch.Tensor,
        pca_mean: torch.Tensor,
        pca_components: torch.Tensor,
        u_grid: torch.Tensor,
        numvox: torch.Tensor,
        min_xyz: torch.Tensor,
        max_xyz: torch.Tensor,
        log_quantile_C: float = 1e-2,
        t_max_ns: float = 600.0,
        mode: str = "log_quantile",
        n_simulated: float = DEFAULT_N_SIMULATED,
        device: str = "cpu",
        interpolate: bool = True,
        pmt_qe: float = 1.0,
    ) -> "TOFSampler":
        """Construct a TOFSampler from in-memory arrays, bypassing h5 file loading.

        Useful for tests and synthetic libraries.  Shapes:
          vis: (n_voxels, n_pmts), t0: (n_voxels, n_pmts),
          coeffs: (n_voxels, n_pmts, n_components),
          pca_mean: (Q,), pca_components: (K, Q), u_grid: (Q,),
          numvox: (3,), min_xyz: (3,), max_xyz: (3,)
        """
        inst = cls.__new__(cls)
        dev = torch.device(device)
        inst._lazy = False
        inst._interpolate = interpolate
        inst._file = None
        inst._init_common(
            device=dev,
            n_simulated=n_simulated,
            pmt_qe=pmt_qe,
            n_pmts=vis.shape[1],
            n_components=coeffs.shape[2],
            log_quantile_C=log_quantile_C,
            t_max_ns=t_max_ns,
            mode=mode,
            pca_mean=pca_mean,
            pca_components=pca_components,
            u_grid=u_grid,
            numvox=numvox,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
        )
        inst._n_voxels = int(vis.shape[0])
        inst.vis = vis.to(dtype=torch.float32, device=dev)
        inst.t0 = t0.to(dtype=torch.float32, device=dev)
        inst.coeffs = coeffs.to(dtype=torch.float32, device=dev)
        return inst

    # LUT fetch (trilinear or nearest-neighbor)

    def _coord_to_voxel(self, pos):
        """pos: (N, 3) -> (N,) voxel indices via flat raveled index."""
        pos = pos.to(dtype=torch.float64, device=self._device)  # (N, 3)
        frac = (pos - self._min_xyz) / (self._max_xyz - self._min_xyz + 1e-12)  # (N, 3) normalized [0,1]
        idx = (frac * self._numvox.double()).long().clamp(min=0)  # (N, 3) integer grid coords
        for d in range(3):
            idx[:, d] = idx[:, d].clamp(max=self._numvox[d] - 1)
        nx, ny, nz = self._numvox
        return idx[:, 0] + nx * (idx[:, 1] + ny * idx[:, 2])  # (N,) flat voxel id

    def _fetch(self, voxel_ids):
        """Fetch vis, t0, coeffs for voxel_ids.
        voxel_ids: (N,) -> vis: (N, P), t0: (N, P), coeffs: (N, P, K)
        """
        if self.vis is not None:
            return self.vis[voxel_ids], self.t0[voxel_ids], self.coeffs[voxel_ids]
        ids_np = voxel_ids.cpu().numpy()
        uniq, inv = np.unique(ids_np, return_inverse=True)
        v = torch.from_numpy(self._file["vis"][uniq]).float()      # (U, P)
        t = torch.from_numpy(self._file["t0"][uniq]).float()       # (U, P)
        c = torch.from_numpy(self._file["coeffs"][uniq]).float()   # (U, P, K)
        inv_t = torch.from_numpy(inv).long()                       # (N,) maps back to uniq
        return v[inv_t].to(self._device), t[inv_t].to(self._device), c[inv_t].to(self._device)

    def _trilinear_fetch(self, pos):
        """pos: (N, 3) -> interpolated (vis, t0, coeffs) via trilinear blending of 8 corner voxels."""
        pos = pos.to(dtype=torch.float64, device=self._device)
        frac = (pos - self._min_xyz) / (self._max_xyz - self._min_xyz + 1e-12)
        cont = frac * self._numvox.double() - 0.5  # continuous index, voxel-center-aligned

        idx0 = cont.long().clamp(min=0)  # floor corner (N, 3)
        idx1 = idx0 + 1                  # ceil corner (N, 3)
        for d in range(3):
            idx0[:, d].clamp_(max=self._numvox[d] - 1)
            idx1[:, d].clamp_(max=self._numvox[d] - 1)

        w = (cont - idx0.double()).clamp(0, 1).float()  # (N, 3) fractional weights

        # 8 corners: compute voxel IDs and trilinear weights
        corners = []   # list of 8 (N,) voxel ID tensors
        weights = []   # list of 8 (N,) scalar weight tensors
        nx, ny, nz = self._numvox
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ix = idx1[:, 0] if dx else idx0[:, 0]
                    iy = idx1[:, 1] if dy else idx0[:, 1]
                    iz = idx1[:, 2] if dz else idx0[:, 2]
                    corners.append(ix + nx * (iy + ny * iz))
                    wx = w[:, 0] if dx else (1 - w[:, 0])
                    wy = w[:, 1] if dy else (1 - w[:, 1])
                    wz = w[:, 2] if dz else (1 - w[:, 2])
                    weights.append(wx * wy * wz)

        # Batch fetch all 8*N voxel IDs (dedup inside _fetch)
        N = pos.shape[0]
        all_vox = torch.cat(corners)  # (8N,)
        all_vis, all_t0, all_coeffs = self._fetch(all_vox)

        # Reshape to (8, N, ...) and blend with weights (8, N, 1)
        w8 = torch.stack(weights).unsqueeze(-1)                  # (8, N, 1)
        vis = (w8 * all_vis.view(8, N, -1)).sum(0)               # (N, P)
        t0 = (w8 * all_t0.view(8, N, -1)).sum(0)                 # (N, P)
        coeffs = (w8.unsqueeze(-1) * all_coeffs.view(8, N, self._n_pmts, -1)).sum(0)  # (N, P, K)

        return vis, t0, coeffs

    def _lookup(self, pos):
        """Dispatch to trilinear or nearest-neighbor fetch based on self._interpolate."""
        if self._interpolate:
            return self._trilinear_fetch(pos)
        vox = self._coord_to_voxel(pos)
        return self._fetch(vox)

    # ---- lifecycle override: close h5 handle -----------------------------

    def close(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None


# Back-compat alias: ``sample_pdf`` is now available on every ``PCATOFSampler``
# (including the regular ``TOFSampler``). External callers that reference
# ``DifferentiableTOFSampler`` continue to work unchanged.
DifferentiableTOFSampler = TOFSampler


class QuantileTOFSampler(QuantileReconMixin, TOFSampler):
    """Voxel-LUT sampler over a QUANTILE-native photon library.

    The third leg of the SIREN-vs-LUT comparison: where ``TOFSampler`` reads a
    PCA-compressed plib (`coeffs` + `pca_components`) and reconstructs the quantile
    function from 50 components, this reads the stored quantile function directly.

    Differences from the parent, all forced by the quantile file layout:
      * the per-voxel tensor is ``quantiles`` (n_voxels, P, 512), not ``coeffs``
      * t0 lives in ``analytical_t0`` and may be stored as integer TICKS rather than ns
        it is converted here using tick = t_max_ns / n_bins
      * the quantile axis is strided by ``combine_every_quantile``

    Trilinear blending happens in the stored (log-quantile) domain, before the 10**x
    decode.
    """

    def __init__(
        self,
        filepath,
        n_simulated=DEFAULT_N_SIMULATED,
        device="cpu",
        interpolate=True,
        pmt_qe=None,
        combine_every_quantile=1,
        verbose=True,
    ):
        dev = torch.device(device)
        self._lazy = True          # not negotiable, see class docstring
        self._interpolate = interpolate
        self._file = None
        self._cq = max(1, int(combine_every_quantile))

        basis = PCATOFSampler._read_h5_basis(filepath, combine_every_quantile=self._cq)
        if basis["has_pca"]:
            # Not an error -- a PCA plib also carries `quantiles` in some productions --
            # but the user probably wanted the PCA leg, so say which one they got.
            if verbose:
                print(
                    "[QuantileTOFSampler] note: this file also has a PCA basis; reading "
                    "the raw `quantiles` dataset and ignoring it. Use TOFSampler for the "
                    "PCA-reconstructed leg."
                )
        self._init_common(
            device=dev,
            n_simulated=n_simulated,
            pmt_qe=float(pmt_qe) if pmt_qe is not None else 1.0,
            n_pmts=basis["n_pmts"],
            n_components=basis["n_components"],   # = Q after striding
            log_quantile_C=basis["log_quantile_C"],
            t_max_ns=basis["t_max_ns"],
            mode=basis["mode"],
            pca_mean=None,          # quantile-native: no basis to expand against
            pca_components=None,
            u_grid=basis["u_grid"],
            numvox=basis["numvox"],
            min_xyz=basis["min_xyz"],
            max_xyz=basis["max_xyz"],
        )
        self._n_voxels = basis["n_voxels"]
        self.vis = self.t0 = self.coeffs = None   # lazy: always fetched from the file

        self._file = h5py.File(filepath, "r", swmr=True, libver="latest")
        if "quantiles" not in self._file:
            raise KeyError(f"{filepath} has no 'quantiles' dataset")
        self._t0_dset = "analytical_t0" if "analytical_t0" in self._file else "t0"

        # t0 units: sirentv treats t0 as ns when the `t0_in_ns` attr is set, when mode is
        # "quantile", or when the stored dtype is floating point; otherwise it is an
        # integer tick index that must be scaled by the bin width.
        n_bins = int(self._file.attrs.get("n_bins", 1000))
        self._tick_ns = float(basis["t_max_ns"]) / max(1, n_bins)
        self._t0_in_ns = bool(
            self._file.attrs.get("t0_in_ns", False)
            or basis["mode"] == "quantile"
            or self._file[self._t0_dset].dtype.kind == "f"
        )
        if verbose:
            print(
                f"[QuantileTOFSampler] Q={basis['n_components']} "
                f"(combine_every_quantile={self._cq}), mode={basis['mode']}, "
                f"t0 from '{self._t0_dset}' in "
                f"{'ns' if self._t0_in_ns else f'ticks x {self._tick_ns:.4g} ns'}"
            )

    def _fetch(self, voxel_ids):
        """voxel_ids: (N,) -> vis (N, P), t0_ns (N, P), quantiles (N, P, Q).

        Mirrors the parent's lazy path (unique + fancy-index + invert) but reads the
        quantile tensor, strides it, and converts t0 to ns.
        """
        ids_np = voxel_ids.cpu().numpy()
        uniq, inv = np.unique(ids_np, return_inverse=True)
        v = torch.from_numpy(self._file["vis"][uniq]).float()                 # (U, P)
        t_raw = self._file[self._t0_dset][uniq]                               # (U, P)
        q = torch.from_numpy(
            self._file["quantiles"][uniq][:, :, ::self._cq]
        ).float()                                                             # (U, P, Q)

        t = torch.from_numpy(np.asarray(t_raw)).float()
        if not self._t0_in_ns:
            t = t.clamp(min=0) * self._tick_ns

        inv_t = torch.from_numpy(inv).long()
        return (
            v[inv_t].to(self._device),
            t[inv_t].to(self._device),
            q[inv_t].to(self._device),
        )


def create_quantile_tof_sampler(**kwargs) -> QuantileTOFSampler:
    """Factory for the quantile-LUT leg. `plib_path` defaults to the site's quantile
    library; pass `combine_every_quantile`."""
    from ..sites import paths as _site_paths

    plib_path = kwargs.pop("plib_path", kwargs.pop("filepath", None))
    if not plib_path:
        plib_path = _site_paths()["quantile_plib"]
        if not plib_path:
            raise ValueError(
                "no quantile library configured for this site -- pass plib_path= or add "
                "one to goop/sites.py"
            )
    defaults = {
        "n_simulated": DEFAULT_N_SIMULATED,
        "device": "cuda:0",
        "interpolate": True,
        "pmt_qe": 0.12,
    }
    defaults.update(kwargs)
    if "combine_every_quantile" not in kwargs:
        defaults["combine_every_quantile"] = _q3_stride_or_1()
    return QuantileTOFSampler(plib_path, **defaults)


def _q3_stride_or_1():
    """photonlib.combine_every_quantile from the site's train_cfg.yaml, else 1."""
    import yaml
    from ..sites import paths as _site_paths

    cfg_path = _site_paths().get("train_cfg")
    if not cfg_path:
        return 1
    try:
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
    except OSError:
        return 1
    plib_cfg = cfg.get("compressed_plib", cfg.get("photonlib", {})) or {}
    return int(plib_cfg.get("combine_every_quantile", 1))
