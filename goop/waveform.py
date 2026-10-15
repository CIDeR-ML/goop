"""Waveform types for the optical TPC simulation pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .digitize import digitize as _digitize
from .waveform_utils import _gather_rows, _next_fft_size, _slice_channel

# Largest batch, in elements, that SlicedWaveform.convolve (rows x FFT size) and
# .downsample (fine bins) process at once; bounds their temporary buffers.
_BATCH_ELEMENTS = 1 << 25


@dataclass
class Waveform:
    """Per-channel 1D data on a shared global time axis.

    All channels have the same length and share a single time origin (t0).
    Used for both pre-convolution histograms and post-convolution waveforms.
    """

    adc: torch.Tensor   # (n_channels, n_bins)
    t0: float           # global time origin in ns
    tick_ns: float      # time step in ns
    n_channels: int     # number of PMT channels
    attrs: dict = field(default_factory=dict)  # arbitrary metadata (e.g. pe_counts)

    @staticmethod
    def from_photons(
        times: torch.Tensor,
        channels: torch.Tensor,
        tick_ns: float,
        n_channels: int,
        t0: float = None,
        t0_snap_ns: float = None,
        weights: torch.Tensor = None,
    ) -> Waveform:
        """Build per-channel histograms on a shared global time axis.

        NOTE: this materializes the full global histogram in memory.
        For events with huge time ranges, this may cause an OOM.

        Parameters
        ----------
        t0 : Global time origin in ns. Defaults to times.min(),
             snapped to the tick boundary.
        t0_snap_ns : Grid spacing for t0 snapping (default: tick_ns).
             Use a coarser value when oversampling to keep t0 on the
             output grid.
        weights : Per-photon weights (default: 1.0 per photon).
        """
        device = times.device

        if times.numel() == 0:
            return Waveform(
                adc=torch.zeros(n_channels, 1, device=device),
                t0=t0 or 0.0, tick_ns=tick_ns, n_channels=n_channels,
            )
        if t0 is None:
            snap = t0_snap_ns or tick_ns
            t0 = (times.min() / snap).floor().item() * snap
        shifted = times - t0
        if (shifted < 0).any():
            raise ValueError(
                f"Some photon times are before t0={t0:.2f} ns "
            )
        n_bins = int(shifted.max().item() / tick_ns) + 1

        data = torch.zeros(n_channels, n_bins, device=device, dtype=torch.float32)
        bin_idx = (shifted / tick_ns).long().clamp(max=n_bins - 1)

        flat_idx = channels.long() * n_bins + bin_idx
        w = weights if weights is not None else torch.ones_like(flat_idx, dtype=torch.float32)
        data.view(-1).scatter_add_(0, flat_idx, w)

        return Waveform(adc=data, t0=t0, tick_ns=tick_ns, n_channels=n_channels)

    def slice(self, kernel_extent_ns: float) -> SlicedWaveform:
        """Compress by removing dead gaps (zero-stretches > kernel_extent_ns)."""
        kernel_extent_bins = int(kernel_extent_ns / self.tick_ns)
        device = self.adc.device
        all_adc: List[torch.Tensor] = []
        offsets_list = [0]
        all_t0: List[float] = []
        all_pmt: List[int] = []

        for ch in range(self.n_channels):
            compressed, bin_starts, t0_starts = _slice_channel(
                self.adc[ch], self.t0, self.tick_ns, kernel_extent_bins
            )
            n_ch_chunks = len(bin_starts)
            for k in range(n_ch_chunks):
                c0 = bin_starts[k]
                c1 = bin_starts[k + 1] if k + 1 < n_ch_chunks else compressed.numel()
                all_adc.append(compressed[c0:c1])
                offsets_list.append(offsets_list[-1] + (c1 - c0))
                all_t0.append(t0_starts[k])
                all_pmt.append(ch)

        return SlicedWaveform(
            adc=torch.cat(all_adc) if all_adc else torch.zeros(0, device=device),
            offsets=torch.tensor(offsets_list, device=device, dtype=torch.long),
            t0_ns=torch.tensor(all_t0, device=device, dtype=torch.float32),
            pmt_id=torch.tensor(all_pmt, device=device, dtype=torch.long),
            tick_ns=self.tick_ns,
            n_channels=self.n_channels,
            n_bins=self.adc.shape[1],
            attrs=dict(self.attrs),
        )

    def digitize(self, pedestal: float, n_bits: int) -> Waveform:
        """Apply pedestal, quantize, and saturate to ADC range."""
        return Waveform(
            adc=_digitize(self.adc, pedestal, n_bits),
            t0=self.t0, tick_ns=self.tick_ns, n_channels=self.n_channels,
            attrs=dict(self.attrs),
        )

    def deslice(self, fill: Optional[float] = None) -> Waveform:
        """No-op on a dense ``Waveform`` — returns ``self``.

        Provided for interface parity with ``SlicedWaveform.deslice()`` so
        downstream code doesn't need to care which kind of waveform the
        simulator produced (e.g. the streaming diff-sim returns a dense
        ``Waveform`` directly rather than a ``SlicedWaveform``).
        """
        return self

    def convolve(self, kernel: torch.Tensor, gain: float) -> Waveform:
        """FFT-convolve all channels with kernel and apply gain."""
        conv_ticks = kernel.shape[0]
        n_ch, n_tick = self.adc.shape
        n_fft = _next_fft_size(n_tick + conv_ticks - 1)

        padded = F.pad(self.adc, (0, n_fft - n_tick))
        k_fft = torch.fft.rfft(kernel, n=n_fft)
        result = gain * torch.fft.irfft(
            torch.fft.rfft(padded) * k_fft.unsqueeze(0), n=n_fft
        )
        out_len = n_tick + conv_ticks - 1
        return Waveform(
            adc=result[:, :out_len], t0=self.t0,
            tick_ns=self.tick_ns, n_channels=self.n_channels,
            attrs=dict(self.attrs),
        )

    def downsample(self, factor: int) -> Waveform:
        """Average groups of `factor` consecutive bins into coarser bins."""
        if factor == 1:
            return self
        n_ch, n_bins = self.adc.shape
        remainder = n_bins % factor
        data = self.adc
        if remainder != 0:
            data = F.pad(data, (0, factor - remainder))
        n_coarse = data.shape[1] // factor
        coarse = data.reshape(n_ch, n_coarse, factor).mean(dim=2)
        return Waveform(
            adc=coarse, t0=self.t0,
            tick_ns=self.tick_ns * factor, n_channels=self.n_channels,
            attrs=dict(self.attrs),
        )

    def align_to(self, t0: float, n_bins: int, fill: float = 0.0) -> Waveform:
        """Pad/crop this waveform onto the (t0, n_bins) grid.

        Autograd-safe: gradients flow back through `self.adc` for any sample
        that survives the crop.
        """
        offset = int(round((self.t0 - t0) / self.tick_ns))
        out = torch.full(
            (self.n_channels, n_bins), fill,
            device=self.adc.device, dtype=self.adc.dtype,
        )
        src_start = max(0, -offset)
        dst_start = max(0, offset)
        n_copy = min(self.adc.shape[1] - src_start, n_bins - dst_start)
        if n_copy > 0:
            out[:, dst_start:dst_start + n_copy] = (
                self.adc[:, src_start:src_start + n_copy]
            )
        return Waveform(
            adc=out, t0=float(t0),
            tick_ns=self.tick_ns, n_channels=self.n_channels,
            attrs=dict(self.attrs),
        )

    def align_with(
        self, other: Waveform, fill: float = 0.0,
    ) -> Tuple[Waveform, Waveform]:
        """Pad both waveforms onto their union (t0, n_bins) grid."""
        if abs(self.tick_ns - other.tick_ns) > 1e-9:
            raise ValueError(
                f"tick_ns mismatch: {self.tick_ns} vs {other.tick_ns}"
            )
        tick = self.tick_ns
        t0 = min(self.t0, other.t0)
        n_bins = max(
            int(round((self.t0 - t0) / tick)) + self.adc.shape[1],
            int(round((other.t0 - t0) / tick)) + other.adc.shape[1],
        )
        return self.align_to(t0, n_bins, fill), other.align_to(t0, n_bins, fill)


@dataclass
class SlicedWaveform:
    """Compressed waveform stored as flat CSR-style arrays.

    Each chunk is a contiguous segment of ADC data tagged with a PMT channel
    and a real-time origin.  Chunks are ordered by (pmt_id, t0_ns).

    Fields
    ------
    adc      : (total_bins,)  all chunks concatenated
    offsets  : (n_chunks+1,)  CSR boundaries — chunk k = adc[offsets[k]:offsets[k+1]]
    t0_ns    : (n_chunks,)    real-time start of each chunk in ns
    pmt_id   : (n_chunks,)    PMT channel of each chunk
    tick_ns  : time bin width in ns
    n_channels : total number of PMT channels
    """

    adc: torch.Tensor       # (total_bins,)
    offsets: torch.Tensor   # (n_chunks+1,)
    t0_ns: torch.Tensor    # (n_chunks,)
    pmt_id: torch.Tensor    # (n_chunks,)
    tick_ns: float
    n_channels: int
    n_bins: int = None       # expected desliced length (optional, set by slice/convolve)
    attrs: dict = field(default_factory=dict)  # arbitrary metadata (e.g. pe_counts)

    @property
    def n_chunks(self) -> int:
        return self.pmt_id.numel()

    def chunk(self, k: int) -> torch.Tensor:
        """Return ADC data for chunk k."""
        return self.adc[self.offsets[k]:self.offsets[k + 1]]

    @staticmethod
    def from_photons(
        times: torch.Tensor,
        channels: torch.Tensor,
        tick_ns: float,
        n_channels: int,
        kernel_extent_ns: float,
        t0_snap_ns: float = None,
        weights: torch.Tensor = None,
    ) -> SlicedWaveform:
        """Build compressed per-channel histograms directly from photon times.

        Parameters
        ----------
        t0_snap_ns : Grid spacing for t0 snapping (default: tick_ns).
             Use a coarser value when oversampling to keep chunk t0
             values on the output grid.
        weights : Per-photon weights (default: 1.0 per photon).
        """
        snap = t0_snap_ns or tick_ns
        device = times.device

        # Each channel's photons as one run, in their input order (channels
        # outside [0, n_channels) are ignored); each run is sorted below.
        keep = (channels >= 0) & (channels < n_channels)
        ch, by_ch = channels[keep].sort(stable=True)
        t = times[keep][by_ch]
        w = weights[keep][by_ch] if weights is not None else torch.ones_like(t, dtype=torch.float32)
        counts = torch.bincount(ch, minlength=n_channels).tolist()
        active = [c for c in range(n_channels) if counts[c] > 0]
        if not active:
            return SlicedWaveform(
                adc=torch.zeros(0, device=device),
                offsets=torch.zeros(1, device=device, dtype=torch.long),
                t0_ns=torch.zeros(0, device=device, dtype=torch.float32),
                pmt_id=torch.zeros(0, device=device, dtype=torch.long),
                tick_ns=tick_ns,
                n_channels=n_channels,
            )
        ends = torch.tensor(counts, device=device).cumsum(0)
        firsts = (ends - torch.tensor(counts, device=device))[active]
        lasts = ends[active] - 1

        # Per channel: sort by time and cut out dead time beyond the kernel
        # extent. Each channel gets its own sort and cumsum, as before, so ties
        # and the running sum come out the same.
        cum_dead = torch.zeros_like(t)
        for a, b in zip(firsts.tolist(), (lasts + 1).tolist()):
            t[a:b], o = t[a:b].sort()
            w[a:b] = w[a:b][o]
            cum_dead[a + 1:b] = (torch.diff(t[a:b]) - kernel_extent_ns).clamp(min=0).cumsum(0)
        diffs = torch.diff(t)
        ch_t0 = torch.zeros(n_channels, device=device, dtype=t.dtype)
        ch_t0[active] = (t[firsts] / snap).floor() * snap
        compressed_t = t - cum_dead - ch_t0[ch]

        # One histogram per channel, laid end to end.
        n_bins = torch.zeros(n_channels, device=device, dtype=torch.long)
        n_bins[active] = torch.tensor([int(v / tick_ns) + 1 for v in compressed_t[lasts].tolist()], device=device)
        hist_start = n_bins.cumsum(0) - n_bins
        bin_idx = torch.minimum((compressed_t / tick_ns).long(), n_bins[ch] - 1)
        adc = torch.zeros(int(n_bins.sum()), device=device, dtype=torch.float32)
        adc.scatter_add_(0, hist_start[ch] + bin_idx, w)

        # Chunks: each channel's bin 0, then one per gap longer than the kernel
        # extent. A post-gap chunk's t0 is the absolute time of its bin 0, on
        # the fine `tick_ns` grid (NOT `snap`): snapping it to a coarser grid
        # would break the alignment between bin index and absolute time.
        gap = torch.nonzero((diffs > kernel_extent_ns) & (ch[1:] == ch[:-1])).squeeze(1) + 1
        comp_bins = (compressed_t[gap] / tick_ns).long()
        real_times = ch_t0[ch[gap]] + comp_bins.to(compressed_t.dtype) * tick_ns + cum_dead[gap]
        active_t = torch.tensor(active, device=device)
        starts = torch.cat([hist_start[active_t], hist_start[ch[gap]] + comp_bins])
        chunk_order = starts.sort(stable=True).indices
        return SlicedWaveform(
            adc=adc,
            offsets=torch.cat([starts[chunk_order], adc.new_tensor([adc.numel()], dtype=torch.long)]),
            t0_ns=torch.cat([ch_t0[active_t], real_times])[chunk_order].to(torch.float32),
            pmt_id=torch.cat([active_t, ch[gap]])[chunk_order].long(),
            tick_ns=tick_ns,
            n_channels=n_channels,
        )

    def deslice(self, fill: Optional[float] = None) -> Waveform:
        """Decompress all channels back to a shared global time axis.

        Parameters
        ----------
        fill : float, optional
            Value for dead (non-active) regions.  Defaults to
            `attrs["pedestal"]` if present, otherwise `0.0`.
        """
        if fill is None:
            fill = float(self.attrs.get("pedestal", 0.0))

        device = self.adc.device

        if self.n_chunks == 0:
            # Honor self.n_bins if set (e.g. by an empty convolve preserving
            # the kernel-extent canvas); otherwise default to a single bin.
            n_bins = max(1, self.n_bins) if self.n_bins is not None else 1
            return Waveform(
                adc=torch.full((self.n_channels, n_bins), fill, device=device),
                t0=0.0, tick_ns=self.tick_ns, n_channels=self.n_channels,
                attrs=dict(self.attrs),
            )

        global_t0 = self.t0_ns.min().item()

        chunk_lens = self.offsets[1:] - self.offsets[:-1]             # (n_chunks,)
        start_bins = ((self.t0_ns - global_t0) / self.tick_ns).round().long()  # (n_chunks,)
        if self.n_bins is not None:
            n_bins = self.n_bins
        else:
            n_bins = max(1, int((start_bins + chunk_lens).max().item()))

        # Build a flat destination index for every adc element in one shot.
        # For element i in the flat adc tensor:
        #   chunk k = searchsorted(offsets[1:], i)
        #   within-chunk position j = i - offsets[k]
        #   dst = pmt_id[k] * n_bins + start_bins[k] + j
        total = self.adc.numel()
        flat_i = torch.arange(total, device=device, dtype=torch.long)
        k = torch.searchsorted(self.offsets[1:].contiguous(), flat_i, side="right")
        k = k.clamp(0, self.n_chunks - 1)

        within_chunk = flat_i - self.offsets[k]
        dst_bin = start_bins[k] + within_chunk
        dst_ch = self.pmt_id[k]
        flat_dst = dst_ch * n_bins + dst_bin.clamp(0, n_bins - 1)

        data = torch.full(
            (self.n_channels * n_bins,), fill, device=device, dtype=torch.float32
        )
        data = data.scatter(0, flat_dst, self.adc)
        data = data.view(self.n_channels, n_bins)

        return Waveform(
            adc=data, t0=global_t0,
            tick_ns=self.tick_ns, n_channels=self.n_channels,
            attrs=dict(self.attrs),
        )

    def deslice_channel(self, channel: int, fill: Optional[float] = None) -> Tuple[float, torch.Tensor]:
        """Decompress one channel. Returns (t0_ns, 1D waveform tensor).

        Parameters
        ----------
        fill : float, optional
            Value for dead regions.  Defaults to `attrs["pedestal"]`
            if present, otherwise `0.0`.
        """
        if fill is None:
            fill = float(self.attrs.get("pedestal", 0.0))

        if channel >= self.n_channels:
            raise IndexError(
                f"channel {channel} out of range for {self.n_channels} channels"
            )

        ch_indices = torch.where(self.pmt_id == channel)[0]

        if ch_indices.numel() == 0:
            return 0.0, torch.full((1,), fill, device=self.adc.device, dtype=self.adc.dtype)

        ch_t0 = float('inf')
        ch_t_end = float('-inf')
        for idx in ch_indices:
            k = idx.item()
            chunk_len = int(self.offsets[k + 1] - self.offsets[k])
            t_start = self.t0_ns[k].item()
            t_end = t_start + chunk_len * self.tick_ns
            ch_t0 = min(ch_t0, t_start)
            ch_t_end = max(ch_t_end, t_end)

        n_real = max(1, int((ch_t_end - ch_t0) / self.tick_ns))
        real_wf = torch.full((n_real,), fill, device=self.adc.device, dtype=self.adc.dtype)

        for idx in ch_indices:
            k = idx.item()
            chunk_data = self.adc[self.offsets[k]:self.offsets[k + 1]]
            r0 = int((self.t0_ns[k].item() - ch_t0) / self.tick_ns)
            chunk_len = chunk_data.numel()
            end = min(r0 + chunk_len, n_real)
            real_wf[r0:end] = chunk_data[:end - r0]

        return ch_t0, real_wf

    def digitize(self, pedestal: float, n_bits: int) -> SlicedWaveform:
        """Apply pedestal, quantize, and saturate to ADC range."""
        return SlicedWaveform(
            adc=_digitize(self.adc, pedestal, n_bits),
            offsets=self.offsets.clone(),
            t0_ns=self.t0_ns.clone(),
            pmt_id=self.pmt_id.clone(),
            tick_ns=self.tick_ns,
            n_channels=self.n_channels,
            n_bins=self.n_bins,
            attrs=dict(self.attrs),
        )

    def convolve(self, kernel: torch.Tensor, gain: float) -> SlicedWaveform:
        """FFT-convolve each chunk independently with kernel and apply gain.

        Per-PMT, only the *last* chunk (largest `t0_ns`) keeps the linear-conv
        extension of `conv_ticks - 1` trailing bins (the SER tail of late
        photons). Earlier chunks are truncated to `chunk_data.numel()` so that
        their post-conv extents — which are guaranteed to be zero, since
        `slice()` keeps `kernel_extent_bins` of trailing zeros — don't reach
        into the start of the next chunk for the same PMT and corrupt
        `deslice()`'s scatter (which is non-deterministic on duplicate
        destination indices on CUDA).
        """
        conv_ticks = kernel.shape[0]
        device = self.adc.device
        n_chunks = self.n_chunks
        new_pieces: List[torch.Tensor] = [None] * n_chunks
        new_offsets = [0]

        if n_chunks > 0:
            max_t0_per_pmt = torch.full(
                (self.n_channels,), float("-inf"),
                device=device, dtype=self.t0_ns.dtype,
            )
            max_t0_per_pmt = max_t0_per_pmt.scatter_reduce(
                0, self.pmt_id, self.t0_ns, reduce="amax", include_self=True,
            )
            is_last_per_pmt = self.t0_ns == max_t0_per_pmt[self.pmt_id]
            is_last_cpu = is_last_per_pmt.tolist()
        else:
            is_last_cpu = []

        # Chunks sharing an FFT size are convolved together, a batch of rows at
        # a time; the offsets come to the host once, not per chunk.
        starts = self.offsets.tolist()
        lengths = [b - a for a, b in zip(starts[:-1], starts[1:])]
        out_lens = [n + (conv_ticks - 1 if last else 0) for n, last in zip(lengths, is_last_cpu)]
        n_ffts = [_next_fft_size(n + conv_ticks - 1) for n in lengths]
        for n_fft in sorted(set(n_ffts)):
            k_fft = torch.fft.rfft(kernel, n=n_fft)
            same = [k for k in range(n_chunks) if n_ffts[k] == n_fft]
            rows_per_batch = max(1, _BATCH_ELEMENTS // n_fft)
            for b in range(0, len(same), rows_per_batch):
                batch = same[b:b + rows_per_batch]
                padded = _gather_rows(self.adc, [starts[k] for k in batch], [lengths[k] for k in batch], n_fft)
                result = gain * torch.fft.irfft(torch.fft.rfft(padded) * k_fft, n=n_fft)
                for row, k in enumerate(batch):
                    new_pieces[k] = result[row, :out_lens[k]]
        for out_len in out_lens:
            new_offsets.append(new_offsets[-1] + out_len)

        # An empty input has no signal but the convolution support is still
        # `conv_ticks` bins wide. Pinning n_bins here makes deslice produce
        # a (n_channels, conv_ticks) dense buffer for downstream noise/digit.
        if self.n_chunks == 0:
            new_n_bins = conv_ticks
        else:
            new_n_bins = self.n_bins + conv_ticks - 1 if self.n_bins is not None else None
        return SlicedWaveform(
            adc=torch.cat(new_pieces) if new_pieces else torch.zeros(0, device=device),
            offsets=torch.tensor(new_offsets, device=device, dtype=torch.long),
            t0_ns=self.t0_ns.clone(),
            pmt_id=self.pmt_id.clone(),
            tick_ns=self.tick_ns,
            n_channels=self.n_channels,
            n_bins=new_n_bins,
            attrs=dict(self.attrs),
        )

    def downsample(self, factor: int) -> SlicedWaveform:
        """Average groups of `factor` consecutive fine bins into coarser bins."""
        if factor == 1:
            return self
        coarse_tick = self.tick_ns * factor
        device = self.adc.device

        # Chunks zero-padded to whole coarse bins in one buffer and averaged in
        # groups of `factor`, at most _BATCH_ELEMENTS fine bins at a time.
        starts = self.offsets.tolist()
        lengths = [b - a for a, b in zip(starts[:-1], starts[1:])]
        new_offsets = [0]
        for n in lengths:
            new_offsets.append(new_offsets[-1] + -(-n // factor))
        pieces: List[torch.Tensor] = []
        k = 0
        while k < len(lengths):
            j, size = k + 1, lengths[k]
            while j < len(lengths) and size + lengths[j] <= _BATCH_ELEMENTS:
                size += lengths[j]
                j += 1
            width = (new_offsets[j] - new_offsets[k]) * factor
            if j == k + 1:
                padded = F.pad(self.adc[starts[k]:starts[j]], (0, width - size))
            else:
                padded = torch.zeros(width, device=device, dtype=self.adc.dtype)
                lens = torch.tensor(lengths[k:j], device=device)
                within = torch.arange(size, device=device) - torch.repeat_interleave(
                    torch.tensor(starts[k:j], device=device) - starts[k], lens, output_size=size)
                dest = torch.repeat_interleave(
                    (torch.tensor(new_offsets[k:j], device=device) - new_offsets[k]) * factor, lens,
                    output_size=size)
                padded[dest + within] = self.adc[starts[k]:starts[j]]
            pieces.append(padded.reshape(-1, factor).mean(dim=1))
            k = j

        new_n_bins = math.ceil(self.n_bins / factor) if self.n_bins is not None else None
        return SlicedWaveform(
            adc=torch.cat(pieces) if pieces else torch.zeros(0, device=device),
            offsets=torch.tensor(new_offsets, device=device, dtype=torch.long),
            t0_ns=self.t0_ns.clone(),
            pmt_id=self.pmt_id.clone(),
            tick_ns=coarse_tick,
            n_channels=self.n_channels,
            n_bins=new_n_bins,
            attrs=dict(self.attrs),
        )
    
    def align(self, fill: Optional[float] = None) -> SlicedWaveform:
        """Rewrite each active channel as a single chunk spanning the global
        ``[min_t0, max_t_end]`` window, padding gaps with ``fill``.

        When ``fill`` is omitted, use ``attrs["pedestal"]`` if present,
        otherwise use ``0.0``.

        Vectorized. (180 ms --> 13 ms)
        """
        if fill is None:
            fill = float(self.attrs.get("pedestal", 0.0))

        device = self.adc.device

        if self.n_chunks == 0:
            return SlicedWaveform(
                adc=torch.zeros(0, device=device, dtype=torch.float32),
                offsets=torch.zeros(1, device=device, dtype=torch.long),
                t0_ns=torch.zeros(0, device=device, dtype=torch.float32),
                pmt_id=torch.zeros(0, device=device, dtype=torch.long),
                tick_ns=self.tick_ns, n_channels=self.n_channels,
                n_bins=self.n_bins, attrs=dict(self.attrs),
            )

        chunk_lens = self.offsets[1:] - self.offsets[:-1]                       # (n_chunks,)
        global_t0 = self.t0_ns.min().item()                                     # output metadata only
        start_bins = ((self.t0_ns - global_t0) / self.tick_ns).round().long()   # (n_chunks,)
        n_bins_global = max(1, int((start_bins + chunk_lens).max().item()))

        # torch.unique returns sorted, so torch.searchsorted gives the inverse map.
        active_channels = torch.unique(self.pmt_id)                             # (n_active,)
        n_active = active_channels.numel()
        row_for_chunk = torch.searchsorted(active_channels, self.pmt_id)        # (n_chunks,)

        # Flat destination index for every adc element — same pattern as deslice().
        total = self.adc.numel()
        flat_i = torch.arange(total, device=device, dtype=torch.long)
        k = torch.searchsorted(
            self.offsets[1:].contiguous(), flat_i, side="right",
        ).clamp(0, self.n_chunks - 1)
        within_chunk = flat_i - self.offsets[k]
        dst_bin = (start_bins[k] + within_chunk).clamp(0, n_bins_global - 1)
        flat_dst = row_for_chunk[k] * n_bins_global + dst_bin

        data = torch.full(
            (n_active * n_bins_global,), fill, device=device, dtype=torch.float32,
        )
        data = data.scatter(0, flat_dst, self.adc)   # differentiable in self.adc

        new_offsets = torch.arange(
            0, (n_active + 1) * n_bins_global, n_bins_global,
            device=device, dtype=torch.long,
        )
        new_t0_ns = torch.full(
            (n_active,), global_t0, device=device, dtype=torch.float32,
        )
        return SlicedWaveform(
            adc=data, offsets=new_offsets, t0_ns=new_t0_ns,
            pmt_id=active_channels.to(dtype=torch.long),
            tick_ns=self.tick_ns, n_channels=self.n_channels,
            n_bins=n_bins_global, attrs=dict(self.attrs),
        )
