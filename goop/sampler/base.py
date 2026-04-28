"""
Shared base for PCA-compressed photon-library TOF samplers.

``PCATOFSampler`` holds the quantile-time reconstruction + sampling machinery
(Poisson + inverse-CDF, differentiable PDF deposition) that is independent of
*how* ``(vis, t0, coeffs)`` are produced. Subclasses implement only
``_lookup(pos)``.
"""

from abc import abstractmethod

import h5py
import numpy as np
import torch

from ..base import TOFSamplerBase

__all__ = [
    "PCATOFSampler",
    "DEFAULT_PLIB_PATH",
    "DEFAULT_N_SIMULATED",
]

DEFAULT_PLIB_PATH = "/sdf/data/neutrino/youngsam/compressed_plib_b04_quantile_log_n50.h5"
DEFAULT_N_SIMULATED = 15_000_000


class PCATOFSampler(TOFSamplerBase):
    """Abstract base for PCA-compressed TOF samplers.

    Subclasses implement only ``_lookup(pos) -> (vis, t0, coeffs)`` — how the
    per-PMT visibility, t0, and PCA coefficients are obtained (voxel LUT,
    neural network, ...). Everything else — quantile reconstruction, Poisson +
    inverse-CDF sampling (``sample``), and the differentiable PDF deposition
    path (``sample_pdf``) — is shared here.

    All subclasses must populate the following fields in their ``__init__``
    (directly or via ``_init_common`` / ``_read_h5_basis``):

        ``_device``, ``_n_pmts``, ``_n_components``, ``_pmt_qe``,
        ``n_simulated``, ``_log_quantile_C``, ``_t_max_ns``, ``_mode``,
        ``pca_mean`` (Q,), ``pca_components`` (K, Q), ``u_grid`` (Q,),
        ``_du`` (Q,), ``_numvox`` (3,), ``_min_xyz`` (3,), ``_max_xyz`` (3,).

    Full-detector output: the compressed plib is half-detector (x ≤ 0, P
    PMTs); x > 0 positions are x-mirrored before ``_lookup`` and assigned
    PMT ids P..(2P-1). ``n_channels == 2 * _n_pmts``.
    """

    def _init_common(
        self,
        *,
        device,
        n_simulated,
        pmt_qe,
        n_pmts,
        n_components,
        log_quantile_C,
        t_max_ns,
        mode,
        pca_mean,
        pca_components,
        u_grid,
        numvox,
        min_xyz,
        max_xyz,
    ):
        """Populate every shared PCA-sampler field. Call from subclass __init__."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.n_simulated = float(n_simulated)
        self._pmt_qe = float(pmt_qe)
        self._n_pmts = int(n_pmts)
        self._n_components = int(n_components)
        self._log_quantile_C = float(log_quantile_C)
        self._t_max_ns = float(t_max_ns)
        self._mode = str(mode)
        self.pca_mean = pca_mean.to(dtype=torch.float32, device=self._device)
        self.pca_components = pca_components.to(dtype=torch.float32, device=self._device)
        self.u_grid = u_grid.to(dtype=torch.float32, device=self._device)
        self._du = torch.diff(self.u_grid, prepend=torch.zeros(1, device=self._device))
        self._cumdu = torch.cat([
            torch.zeros(1, device=self._device),
            self._du.cumsum(0),
        ])  # (Q+1,) — CDF values at each u-grid point, with 0 prepended.
        # q_stride=1 keeps all Q u-grid points in sample_pdf. Users can set
        # ``sampler.q_stride = K`` (or pass ``q_stride=K`` to sample_pdf) to
        # subsample every K-th quantile point — cuts the (M, Q) tensors
        # proportionally. ``du`` is recomputed from the subsampled grid so
        # total probability mass is preserved.
        self.q_stride = 1
        self._numvox = numvox.to(dtype=torch.long, device=self._device)
        self._min_xyz = min_xyz.to(dtype=torch.float64, device=self._device)
        self._max_xyz = max_xyz.to(dtype=torch.float64, device=self._device)

    @staticmethod
    def _read_h5_basis(filepath):
        """Read shared PCA-basis + voxel-grid + PMT positions from a compressed plib.

        Returns a dict of ready-to-pass fields plus ``pmt_pos`` (np.ndarray) and
        ``n_voxels`` (int). Does not load the per-voxel LUT tensors (vis/t0/coeffs).
        """
        with h5py.File(filepath, "r") as f:
            return dict(
                n_pmts=int(f["vis"].shape[1]),
                n_components=int(f["coeffs"].shape[2]),
                log_quantile_C=float(f.attrs.get("log_quantile_C", 1e-2)),
                t_max_ns=float(f.attrs.get("t_max_ns", 600.0)),
                mode=str(f.attrs.get("mode", "log_quantile")),
                pca_mean=torch.from_numpy(f["pca_mean"][:]).float(),
                pca_components=torch.from_numpy(f["pca_components"][:]).float(),
                u_grid=torch.from_numpy(f["u_grid"][:]).float(),
                numvox=torch.from_numpy(np.asarray(f["numvox"][:], dtype=np.int64)),
                min_xyz=torch.from_numpy(np.asarray(f["min"][:], dtype=np.float64)),
                max_xyz=torch.from_numpy(np.asarray(f["max"][:], dtype=np.float64)),
                pmt_pos=np.asarray(f["pmt_pos"][:]) if "pmt_pos" in f else None,
                n_voxels=int(f["vis"].shape[0]),
            )

    # abstract lookup

    @abstractmethod
    def _lookup(self, pos: torch.Tensor):
        """Return (vis, t0, coeffs) for each position in ``pos`` (N, 3).

        Shapes: vis (N, P), t0 (N, P), coeffs (N, P, K). Must be differentiable
        with respect to ``pos`` if gradients are desired. Input positions are
        assumed to be on the plib's half-detector side (x <= 0); callers handle
        x-mirroring before invoking this method.
        """

    @property
    def n_channels(self) -> int:
        return self._n_pmts * 2  # full detector (both sides of cathode)

    @property
    def t_max_ns(self) -> float:
        """Width of the per-segment quantile-time window the basis covers."""
        return self._t_max_ns

    # shared helpers used by sample_pdf and _histogram_chunk

    def _resolve_q_stride(self, q_stride):
        """Return ``(q_idx, du_eff)`` for the requested quantile-grid stride.

        Precedence: explicit kwarg > sampler attribute > 1 (no subsample).
        Stride>1 preserves total probability mass by recomputing ``du`` from
        the subsampled grid.
        """
        stride = int(q_stride if q_stride is not None else getattr(self, "q_stride", 1))
        if stride <= 1:
            return None, self._du
        Q_full = self.u_grid.shape[0]
        q_idx = torch.arange(0, Q_full, stride, device=self._device)
        u_sub = self.u_grid[q_idx]
        du_eff = torch.diff(u_sub, prepend=torch.zeros(1, device=self._device))
        return q_idx, du_eff

    def _mirror_x(self, pos_chunk):
        """Mirror x>0 sources into the half-detector basis (cathode symmetry).

        Returns ``(pos_lookup, on_pos_side)``: ``pos_lookup`` is a clone of
        ``pos_chunk`` with x signs flipped on x>0 rows so a single LUT/SIREN
        call handles both halves; ``on_pos_side`` records which rows were
        flipped, for routing the output back to the correct PMT id.
        """
        on_pos_side = pos_chunk[:, 0] > 0
        pos_lookup = pos_chunk.clone()
        pos_lookup[on_pos_side, 0] = -pos_lookup[on_pos_side, 0]
        return pos_lookup, on_pos_side

    def _active_pmt_ids(self, on_pos_side, active_mask):
        """Assemble full-detector PMT ids for ``active_mask`` rows.

        Channels 0..P-1 cover x<=0 sources; channels P..2P-1 cover x>0 sources
        via the cathode-symmetry trick. ``on_pos_side`` (from ``_mirror_x``)
        selects the offset.
        """
        P = self._n_pmts
        C = on_pos_side.shape[0]
        pmt_offset = torch.where(on_pos_side, P, 0).unsqueeze(1).expand(C, P)
        pmt_base = torch.arange(P, device=self._device).unsqueeze(0).expand(C, -1)
        return (pmt_base + pmt_offset)[active_mask]

    # PCA reconstruction

    def _quantile_times(self, coeffs, t0, q_idx=None):
        """Reconstruct absolute quantile times from PCA coefficients.
        coeffs: (M, K), t0: (M,) -> q_abs: (M, Q)
        where M = active pairs, K = PCA components, Q = quantile grid points.

        If ``q_idx`` is given, the PCA basis is subsampled to only the columns
        named by ``q_idx``, so the output is ``(M, len(q_idx))`` and the
        ``(M, Q)`` full matrix is never materialised. This is the memory-saving
        path used by ``sample_pdf`` when ``q_stride > 1``.
        """
        if q_idx is None:
            comp = self.pca_components  # (K, Q)
            mean = self.pca_mean         # (Q,)
        else:
            comp = self.pca_components.index_select(1, q_idx)
            mean = self.pca_mean.index_select(0, q_idx)
        raw = coeffs @ comp + mean  # (M, Q_eff)
        if self._mode == "log_quantile":
            q = (torch.pow(10.0, raw) - self._log_quantile_C).clamp(min=0)
        else:
            q = raw.clamp(min=0)
        return q + t0.unsqueeze(-1)

    def _standardize_inputs(self, pos, n_photons, t_step):
        """Normalise (pos, n_photons, t_step) to float32 tensors on ``self._device``.

        Returns ``(pos, scale, t_step)`` where ``scale = n_photons / n_simulated``
        is a (N,) tensor (gradients flow through scale to ``n_photons`` if it
        is a leaf with ``requires_grad``). ``t_step`` is None iff input was None.
        """
        pos = pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos)
        pos = pos.to(dtype=torch.float32, device=self._device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        N = pos.shape[0]

        if isinstance(n_photons, (int, float)):
            scale = torch.full(
                (N,), float(n_photons) / self.n_simulated, device=self._device
            )
        else:
            nph = n_photons if isinstance(n_photons, torch.Tensor) else torch.as_tensor(n_photons)
            scale = nph.to(dtype=torch.float32, device=self._device) / self.n_simulated

        if t_step is None:
            t_step_t = None
        else:
            t_step_t = t_step if isinstance(t_step, torch.Tensor) else torch.as_tensor(t_step)
            t_step_t = t_step_t.to(dtype=torch.float32, device=self._device)

        return pos, scale, t_step_t

    def _per_chunk(
        self,
        pos_chunk,
        scale_chunk,
        t_step_chunk,
        *,
        stochastic: bool,
        q_idx,
        expected_eps: float = 1e-9,
    ):
        """Shared per-chunk core for both photon and histogram outputs.

        Pipeline (chunked over input positions): mirror x>0 sources, look up
        ``(v, t0, coeffs)``, compute ``expected = v * scale * qe``, then either
        Poisson-sample (``stochastic=True``) or threshold (``stochastic=False``)
        to find active (position, PMT) pairs. Reconstruct quantile times for
        the active pairs and shift by ``t_step``.

        Returns ``(q_abs, weights, active_pmt, pos_local)`` or ``None`` if no
        pairs are active in this chunk.

        - ``q_abs``: (M, Q_eff) absolute quantile times in ns
        - ``weights``: (M,) per-pair weight — Poisson count (stoch, integer-valued
          float) or expected PDF mass (PDF, differentiable wrt ``scale``).
        - ``active_pmt``: (M,) full-detector PMT id in ``[0, 2P)``
        - ``pos_local``: (M,) within-chunk position index (0..C-1)
        """
        P = self._n_pmts
        C = pos_chunk.shape[0]

        pos_lookup, on_pos_side = self._mirror_x(pos_chunk)
        v, t0, coeffs = self._lookup(pos_lookup)               # (C,P), (C,P), (C,P,K)
        expected = v * scale_chunk.unsqueeze(1) * self._pmt_qe  # (C, P)

        if stochastic:
            expected = expected.clamp_(min=0)
            counts = torch.poisson(expected)                    # (C, P) integer-valued float
            active_mask = counts > 0.5
            if not bool(active_mask.any()):
                return None
            weights = counts[active_mask]                       # (M,)
        else:
            active_mask = expected.detach() > expected_eps
            if not bool(active_mask.any()):
                return None
            weights = expected[active_mask]                     # (M,) differentiable

        active_coeffs = coeffs[active_mask]
        active_t0 = t0[active_mask]
        active_pmt = self._active_pmt_ids(on_pos_side, active_mask)
        pos_local = active_mask.nonzero(as_tuple=True)[0]

        q_abs = self._quantile_times(active_coeffs, active_t0, q_idx=q_idx)

        if t_step_chunk is not None:
            t_expanded = t_step_chunk.unsqueeze(1).expand(C, P)
            active_t_emit = t_expanded[active_mask]
            q_abs = q_abs + active_t_emit.unsqueeze(-1)

        return q_abs, weights, active_pmt, pos_local

    def sample_photons(
        self,
        pos,
        n_photons,
        t_step=None,
        *,
        stochastic: bool = True,
        return_source_idx: bool = False,
        chunk_size: int = 20000,
        q_stride: int | None = None,
        expected_eps: float = 1e-9,
    ):
        """Sample per-photon ``(time, channel, weight)`` triples.

        ``stochastic=True``: Poisson counts per (position, PMT) → inverse-CDF
        sampling on the quantile grid. Returned weights are unit (1.0).

        ``stochastic=False``: PDF deposition.

        Parameters
        ----------
        pos, n_photons, t_step : array-like
            Per-segment positions (N, 3), photon yields (N,), and emission
            times (N,) in ns. ``t_step`` may be ``None`` (→ zero shift).
        stochastic : bool, optional
            Pick the Poisson path (default) or the PDF deposition path.
        return_source_idx : bool, optional
            If True, append a 4th element to the result: per-photon (or
            per-ghost-photon) global position index in ``[0, N)``.
        chunk_size, q_stride, expected_eps :
            See ``sample_histogram``. ``q_stride`` is ignored in stochastic
            mode (the inverse-CDF sampler needs the full grid).

        Returns
        -------
        ``(times, channels, weights)`` or
        ``(times, channels, weights, source_idx)`` if ``return_source_idx``.
        """
        pos, scale, t_step = self._standardize_inputs(pos, n_photons, t_step)
        N = pos.shape[0]

        if stochastic:
            q_idx = None
            du_eff = self._du
        else:
            q_idx, du_eff = self._resolve_q_stride(q_stride)

        out_times: list[torch.Tensor] = []
        out_ch: list[torch.Tensor] = []
        out_w: list[torch.Tensor] = []
        out_src: list[torch.Tensor] = []
        u_grid = self.u_grid

        from contextlib import nullcontext
        ctx = torch.no_grad() if stochastic else nullcontext()
        with ctx:
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                res = self._per_chunk(
                    pos[start:end], scale[start:end],
                    t_step[start:end] if t_step is not None else None,
                    stochastic=stochastic, q_idx=q_idx, expected_eps=expected_eps,
                )
                if res is None:
                    continue
                q_abs, w_pair, active_pmt, pos_local = res

                if stochastic:
                    counts = w_pair.long()
                    M = counts.shape[0]
                    T = int(counts.sum().item())
                    if T == 0:
                        continue
                    pair_idx = torch.repeat_interleave(
                        torch.arange(M, device=self._device), counts
                    )
                    u = torch.rand(T, device=self._device)
                    idx = torch.searchsorted(u_grid, u).clamp(1, u_grid.shape[0] - 1)
                    u_lo, u_hi = u_grid[idx - 1], u_grid[idx]
                    t_lo = q_abs[pair_idx, idx - 1]
                    t_hi = q_abs[pair_idx, idx]
                    frac = ((u - u_lo) / (u_hi - u_lo + 1e-12)).clamp_(0, 1)
                    t_samp = t_lo + frac * (t_hi - t_lo)

                    out_times.append(t_samp)
                    out_ch.append(active_pmt[pair_idx])
                    out_w.append(torch.ones(T, device=self._device))
                    if return_source_idx:
                        out_src.append(pos_local[pair_idx] + start)
                else:
                    Q = q_abs.shape[1]
                    out_times.append(q_abs.reshape(-1))
                    out_ch.append(active_pmt.unsqueeze(-1).expand(-1, Q).reshape(-1))
                    out_w.append((w_pair.unsqueeze(-1) * du_eff.unsqueeze(0)).reshape(-1))
                    if return_source_idx:
                        out_src.append(
                            (pos_local + start).unsqueeze(-1).expand(-1, Q).reshape(-1)
                        )

        if not out_times:
            empty_f = torch.zeros(0, device=self._device)
            empty_l = torch.zeros(0, device=self._device, dtype=torch.long)
            if return_source_idx:
                return empty_f, empty_l, empty_f, empty_l
            return empty_f, empty_l, empty_f

        times = torch.cat(out_times)
        channels = torch.cat(out_ch)
        weights = torch.cat(out_w)
        if return_source_idx:
            return times, channels, weights, torch.cat(out_src)
        return times, channels, weights

    def sample_histogram(
        self,
        pos,
        n_photons,
        t_step=None,
        *,
        stochastic: bool = True,
        tick_ns: float = 1.0,
        n_bins: int | None = None,
        t0_ref: float = 0.0,
        t_max_ns: float | None = None,
        chunk_size: int = 20000,
        q_stride: int | None = None,
        expected_eps: float = 1e-9,
        use_checkpoint: bool = True,
        label_offsets: torch.Tensor | None = None,
    ):
        """Scatter active-pair quantile times into a dense histogram.

        ``stochastic=True``: Poisson counts spread across the quantile grid by
        ``du`` weights, summed per time bin. Returns ``int32``. Equivalent to
        the old ``sample(return_histogram=True)``.

        ``stochastic=False``: Differentiable PDF deposition — ``expected * du``
        scattered into the histogram. Returns ``float32``. Per-chunk gradient
        checkpointing (``use_checkpoint=True``) keeps peak memory ``O(n_bins)``
        regardless of input segment count.

        ``label_offsets``: optional ``(N,)`` long tensor that shifts each
        segment's PMT writes by ``label_offsets[i] * 2P`` channels. The
        returned histogram has shape ``((max(label_offsets)+1) * 2P, n_bins)``,
        with each label occupying a disjoint channel range. Used to produce
        per-label virtual-channel histograms in a single batched call.
        """
        pos, scale, t_step = self._standardize_inputs(pos, n_photons, t_step)
        N = pos.shape[0]
        P = self._n_pmts
        n_pmts_full = 2 * P

        if n_bins is None:
            window = float(t_max_ns) if t_max_ns is not None else self._t_max_ns
            n_bins = int(round(window / tick_ns))

        if stochastic:
            q_idx = None
            du_eff = self._du
        else:
            q_idx, du_eff = self._resolve_q_stride(q_stride)

        if label_offsets is not None:
            label_offsets = label_offsets.to(dtype=torch.long, device=self._device)
            n_outputs = int(label_offsets.max().item()) + 1 if label_offsets.numel() > 0 else 1
        else:
            n_outputs = 1
        n_ch = n_outputs * n_pmts_full
        inv_tick = 1.0 / float(tick_ns)
        t0_ref = float(t0_ref)

        def _do_chunk(start: int, end: int) -> torch.Tensor:
            chunk_t = t_step[start:end] if t_step is not None else None
            res = self._per_chunk(
                pos[start:end], scale[start:end], chunk_t,
                stochastic=stochastic, q_idx=q_idx, expected_eps=expected_eps,
            )
            chunk_hist = torch.zeros(
                n_ch, n_bins, device=self._device, dtype=torch.float32,
            )
            if res is None:
                return chunk_hist
            q_abs, w_pair, active_pmt, pos_local = res

            if label_offsets is not None:
                chunk_offsets = label_offsets[start:end]
                active_offsets = chunk_offsets[pos_local]
                active_pmt = active_pmt + active_offsets * n_pmts_full

            weights = w_pair.unsqueeze(-1) * du_eff.unsqueeze(0)  # (M, Q)

            bin_idx = ((q_abs - t0_ref) * inv_tick).long()
            in_window = (bin_idx >= 0) & (bin_idx < n_bins)
            bin_idx = bin_idx.clamp(0, n_bins - 1)
            weights = weights * in_window

            flat_idx = active_pmt.unsqueeze(-1) * n_bins + bin_idx
            flat_hist = chunk_hist.view(-1).scatter_add(
                0, flat_idx.reshape(-1), weights.reshape(-1),
            )
            return flat_hist.view(n_ch, n_bins)

        from contextlib import nullcontext
        ctx = torch.no_grad() if stochastic else nullcontext()

        hist = torch.zeros(n_ch, n_bins, device=self._device, dtype=torch.float32)
        with ctx:
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                if not stochastic and use_checkpoint:
                    from torch.utils.checkpoint import checkpoint
                    chunk_hist = checkpoint(_do_chunk, start, end, use_reentrant=False)
                else:
                    chunk_hist = _do_chunk(start, end)
                hist = hist + chunk_hist

        if stochastic:
            return hist.to(torch.int32)
        return hist

    def close(self):
        """No-op by default; LUT subclass overrides to close its h5 handle."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
