"""
SIREN-backed TOF sampler.

Drop-in replacement for the voxel-LUT samplers: quantile-time reconstruction,
Poisson sampling, and the differentiable ``sample_pdf`` path are all inherited
from ``PCATOFSampler``. Only ``_lookup(pos)`` is reimplemented — instead of
trilinear LUT interpolation, a batch of (position, PMT-position) pairs is
forwarded through a pre-trained sirentv network whose time head emits the
quantile function directly (the q3 model: ``{v, t0, quantiles}``).

``QuantileReconMixin`` skips the PCA expansion, so this decodes exactly like
``QuantileTOFSampler`` does the stored library: the two differ only in whether
the numbers come from a network or a LUT.

Differentiability: the network runs in ``eval`` mode with frozen
parameters, but its forward is *not* wrapped in ``torch.no_grad`` —
activations stay in the autograd graph so gradients flow from waveforms
back through ``_lookup`` to input positions.

Paths default per site (NERSC / s3df) from ``goop.sites``; see that module.
"""

from __future__ import annotations

import contextlib
import copy
import os
import re
import sys

import h5py
import numpy as np
import torch
import yaml

from .base import DEFAULT_N_SIMULATED, PCATOFSampler, QuantileReconMixin

_nullctx = contextlib.nullcontext

__all__ = [
    "SirenTOFSampler",
    "create_siren_tof_sampler",
]


def _site_default(key, fallback=""):
    """Resolve a path from goop.sites without failing on an unrecognized machine."""
    try:
        from ..sites import paths
        return paths().get(key) or fallback
    except Exception:
        return fallback


# Resolved at import for `from goop.sampler import DEFAULT_*`. The sampler itself
# re-resolves any path left as None at call time, so goop.sites.set_site() works
# after import.
DEFAULT_CKPT_PATH = _site_default("q3_ckpt_dir")
DEFAULT_CFG_PATH = _site_default("q3_cfg")
DEFAULT_SIRENTV_SRC = _site_default("sirentv_src")

_CKPT_PREFIXES = (
    "_orig_mod.model.",
    "_orig_mod.",
    "model.model.",
    "model.",
)


def _latest_ckpt(path):
    """A checkpoint file as-is, or the highest-epoch ``iteration-*-epoch-*.ckpt`` in a
    run directory (numeric on the epoch, so epoch-900 does not outrank epoch-1000)."""
    if not path:
        raise ValueError("no checkpoint path or directory given")
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} is neither a checkpoint file nor a directory")
    best, best_ep = None, -1
    for name in os.listdir(path):
        m = re.match(r"iteration-(\d+)-epoch-(\d+)\.ckpt$", name)
        if m and int(m.group(2)) > best_ep:
            best, best_ep = os.path.join(path, name), int(m.group(2))
    if best is None:
        raise FileNotFoundError(f"no iteration-*-epoch-*.ckpt found in {path}")
    return best


class SirenTOFSampler(QuantileReconMixin, PCATOFSampler):
    """TOF sampler whose ``_lookup`` is a pre-trained quantile-head SIREN network.

    The network takes a ``(B, P, 6)`` tensor of concatenated normalized
    (source_pos, pmt_pos) pairs and returns ``{v, t0, quantiles}``. The plib is the
    quantile library the network was trained against; its u_grid, strided by the
    training config's ``combine_every_quantile``, is the grid the network's outputs
    are defined on.
    """

    def __init__(
        self,
        plib_path: str | None = None,
        ckpt_path: str | None = None,
        cfg_path: str | None = None,
        sirentv_src: str | None = None,
        n_simulated: float = DEFAULT_N_SIMULATED,
        device: str | torch.device = "cuda:0",
        pmt_qe: float = 0.12,
        n_photon: float | None = None,
        verbose: bool = False,
        autocast_dtype: torch.dtype | None = None,
        use_checkpoint: bool = False,
        site: str | None = None,
        combine_every_quantile: int | None = None,
        strict_load: bool = True,
    ):
        """
        Paths left as None come from goop.sites for ``site`` (auto-detected by
        default). ``ckpt_path`` may be a run directory, in which case the
        highest-epoch checkpoint in it is used -- pass a file to pin one.

        Memory-reduction knobs
        ----------------------
        autocast_dtype : ``torch.bfloat16`` or ``torch.float16`` to run the
            network forward under ``torch.autocast``. Outputs are cast back to
            fp32 before quantile reconstruction (times accumulate into absolute
            ns, which needs fp32 precision). Roughly halves Siren activation
            memory at a small accuracy cost.
        use_checkpoint : if True, wrap ``self.net(inp)`` in
            ``torch.utils.checkpoint.checkpoint`` — only the network's input
            and output are saved for backward; the hidden-layer activations
            are recomputed on the fly, at the cost of one extra forward pass
            during backward.
        """
        dev = torch.device(device) if isinstance(device, str) else device

        # 0. paths
        from ..sites import paths as _site_paths
        sp = _site_paths(site)
        plib_path = plib_path or sp["quantile_plib"]
        cfg_path = cfg_path or sp["q3_cfg"]
        ckpt_path = _latest_ckpt(ckpt_path or sp["q3_ckpt_dir"])
        if sirentv_src is None:
            sirentv_src = sp["sirentv_src"]
        if not plib_path or not cfg_path:
            raise ValueError(
                "no quantile plib / training config for this site; pass plib_path= and "
                "cfg_path= or fill them in goop/sites.py"
            )

        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        plib_cfg = cfg.get("compressed_plib", cfg.get("photonlib", {})) or {}

        # The network emits u_grid[::combine_every_quantile]; take the stride from the
        # training config so the sampler's grid and the network's width agree.
        if combine_every_quantile is None:
            combine_every_quantile = int(plib_cfg.get("combine_every_quantile", 1))
        self._cq = max(1, int(combine_every_quantile))

        # 1. voxel metadata, PMT positions, strided u_grid
        basis = PCATOFSampler._read_h5_basis(plib_path, combine_every_quantile=self._cq)
        if basis["mode"] != "log_quantile":
            raise ValueError(
                f"SirenTOFSampler expects plib mode='log_quantile', got {basis['mode']!r}"
            )
        if basis["pmt_pos"] is None:
            raise ValueError(f"{plib_path} is missing 'pmt_pos' — required for SIREN input")

        self._init_common(
            device=dev,
            n_simulated=n_simulated,
            pmt_qe=pmt_qe,
            n_pmts=basis["n_pmts"],
            n_components=basis["n_components"],
            log_quantile_C=basis["log_quantile_C"],
            t_max_ns=basis["t_max_ns"],
            mode=basis["mode"],
            pca_mean=None,
            pca_components=None,
            u_grid=basis["u_grid"],
            numvox=basis["numvox"],
            min_xyz=basis["min_xyz"],
            max_xyz=basis["max_xyz"],
        )

        # 2. network, built through sirentv's own registry exactly as SirenTV.__init__
        #    does, so the saved config's schema (e.g. `branches:`) is always honoured
        if sirentv_src and sirentv_src not in sys.path:
            sys.path.insert(0, sirentv_src)
        from sirentv.models import build_model  # noqa: E402
        from slar.transform import partial_xform_vis  # noqa: E402

        net_cfg = copy.deepcopy(cfg["model"]["network"])
        net_cfg["xform_vis"] = cfg.get("transform_vis", {}) or {}
        net_cfg["use_CDF"] = str(cfg["model"].get("mode", "pdf")).lower() == "cdf"
        net = build_model(net_cfg)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        clean_state = {}
        for k, v in state.items():
            kk = k
            for prefix in _CKPT_PREFIXES:
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
                    break
            clean_state[kk] = v
        missing, unexpected = net.load_state_dict(clean_state, strict=False)
        if missing or unexpected:
            msg = (
                f"[SirenTOFSampler] load_state_dict from {ckpt_path}: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if missing and strict_load:
                raise RuntimeError(
                    msg + f"\nfirst missing keys: {list(missing)[:5]}\n"
                    "The config and checkpoint do not describe the same architecture. "
                    "Check that cfg_path is the train_cfg.yaml written beside this "
                    "checkpoint, or pass strict_load=False to proceed anyway."
                )
            print(msg)

        net.to(self._device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self._ckpt_path = ckpt_path

        # 3. visibility inverse transform + n_photon
        _, inv_xform_vis = partial_xform_vis(cfg.get("transform_vis", {}))
        self._inv_xform_vis = inv_xform_vis
        if n_photon is None:
            if "n_photon" not in plib_cfg:
                print(
                    "[SirenTOFSampler] warning: no n_photon in the training config "
                    "(looked under compressed_plib and photonlib); falling back to 1.5e7"
                )
            n_photon = plib_cfg.get("n_photon", 1.5e7)
        self._n_photon = float(n_photon)

        # 4. normalized PMT positions, mirroring SirenTV.__init__: a `photonlib` config
        #    reads the pre-normalized `pmt_norm_pos` from the file, a `compressed_plib`
        #    config recomputes them with norm_coord (= _normalize_coord)
        pmt_pos_t = torch.from_numpy(basis["pmt_pos"].astype(np.float32)).to(self._device)
        recomputed = self._normalize_coord(pmt_pos_t)  # (P, 3) in [-1, 1]
        self._norm_pmt_pos = recomputed
        if "compressed_plib" not in cfg and "photonlib" in cfg:
            with h5py.File(plib_path, "r") as f:
                if "pmt_norm_pos" in f:
                    from_file = torch.from_numpy(
                        np.asarray(f["pmt_norm_pos"][:], dtype=np.float32)
                    ).to(self._device)
                    diff = float((from_file - recomputed).abs().max())
                    print(
                        "[SirenTOFSampler] PMT coords: using 'pmt_norm_pos' from the plib, "
                        f"as training did (max |diff| vs recomputed = {diff:.3g})"
                    )
                    self._norm_pmt_pos = from_file
                else:
                    print(
                        "[SirenTOFSampler] warning: training used a photonlib config, "
                        "which reads 'pmt_norm_pos', but this plib has none -- falling "
                        "back to recomputing from min/max"
                    )
        self._plib_path = plib_path

        # 5. memory-reduction knobs
        self.autocast_dtype = autocast_dtype
        self.use_checkpoint = bool(use_checkpoint)

        # 6. a u_grid/network width mismatch would mis-assign every quantile weight
        #    without raising anywhere downstream, so check it once here
        with torch.no_grad():
            x = torch.zeros(1, self._n_pmts, 6, device=self._device)
            width = int(self.net(x)["quantiles"].shape[-1])
        if width != self.u_grid.shape[0]:
            raise ValueError(
                f"network emits {width} quantiles but the u_grid has "
                f"{self.u_grid.shape[0]} points (combine_every_quantile={self._cq}). "
                "Pass the combine_every_quantile the model was trained with, or check "
                "that plib_path is the library it was trained on."
            )

    def _normalize_coord(self, pos: torch.Tensor) -> torch.Tensor:
        """Map world-mm coordinates to [-1, 1] per axis (matches AABox.norm_coord)."""
        lo = self._min_xyz.to(dtype=pos.dtype, device=pos.device)
        hi = self._max_xyz.to(dtype=pos.dtype, device=pos.device)
        return 2.0 * (pos - lo) / (hi - lo) - 1.0

    def _lookup(self, pos: torch.Tensor):
        """pos: (N, 3) on the x<=0 half-detector -> (vis, t0, quantiles).

        Shapes returned: vis (N, P), t0 (N, P), quantiles (N, P, Q) in the raw
        log-quantile domain; QuantileReconMixin decodes them.
        """
        pos = pos.to(dtype=torch.float32, device=self._device)
        N, P = pos.shape[0], self._n_pmts

        pos_norm = self._normalize_coord(pos)                     # (N, 3)
        src = pos_norm.unsqueeze(1).expand(N, P, 3)               # (N, P, 3)
        pmt = self._norm_pmt_pos.unsqueeze(0).expand(N, P, 3)     # (N, P, 3)
        inp = torch.cat([src, pmt], dim=-1)                       # (N, P, 6)

        def _net_forward(x):
            # Return the concatenated (v|t0|quantiles) tensor; checkpoint needs a
            # single tensor output, not a dict.
            d = self.net(x)
            return torch.cat(
                [d["v"].unsqueeze(-1), d["t0"].unsqueeze(-1), d["quantiles"]],
                dim=-1,
            )

        autocast_ctx = (
            torch.autocast(device_type=self._device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else _nullctx()
        )
        with autocast_ctx:
            if self.use_checkpoint and inp.requires_grad:
                from torch.utils.checkpoint import checkpoint
                out = checkpoint(_net_forward, inp, use_reentrant=False)
            else:
                out = _net_forward(inp)

        # Cast back to fp32: downstream quantile reconstruction and absolute-time
        # arithmetic need fp32 precision (times can be microseconds; bf16's
        # 7-bit mantissa gives only ~3 us resolution at 500 us).
        out = out.float()
        v_raw     = out[..., 0]                     # (N, P)
        log_t0    = out[..., 1]                     # (N, P)
        quantiles = out[..., 2:]                    # (N, P, Q)

        vis   = self._inv_xform_vis(v_raw) * self._n_photon
        t0_ns = torch.exp(log_t0)
        return vis, t0_ns, quantiles


def create_siren_tof_sampler(**kwargs) -> SirenTOFSampler:
    """Factory with sensible defaults. See ``SirenTOFSampler.__init__`` for kwargs;
    any path not given is resolved from goop.sites."""
    defaults = {
        "n_simulated": DEFAULT_N_SIMULATED,
        "device": "cuda:0",
        "pmt_qe": 0.12,
    }
    defaults.update(kwargs)
    sampler = SirenTOFSampler(**defaults)
    print(
        f"[goop] SirenTOFSampler: {os.path.basename(sampler._ckpt_path)}, "
        f"Q={sampler.u_grid.shape[0]}, n_pmts={sampler._n_pmts}, "
        f"n_photon={sampler._n_photon:.3g}"
    )
    return sampler
