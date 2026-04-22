import sys
sys.path.append('..')

from goop import (
    SlicedWaveform,
)
from goop.waveform import SlicedWaveform
import torch

def align_interaction(sw: SlicedWaveform, fill: float = 0.0) -> SlicedWaveform:
    device = sw.adc.device
    chunk_lengths = sw.offsets[1:] - sw.offsets[:-1]

    # Exclude chunks with t0=0 when clear real activity exists elsewhere.
    # This handles both explicit placeholders and channels with no photons
    # that were assigned t0=0 by any construction path.
    # Filtering by ADC as well
    nonzero_mask = sw.t0_ns > 0.0
    if nonzero_mask.any():
        min_nonzero_t0 = sw.t0_ns[nonzero_mask].min()

        # Check only the first bin of each chunk
        first_bin_vals = sw.adc[sw.offsets[:-1]]

        # Placeholder: t0 below real activity threshold AND first bin is zero
        is_placeholder = (sw.t0_ns < min_nonzero_t0) & (first_bin_vals == 0.0)
    else:
        is_placeholder = torch.zeros(sw.n_chunks, device=device, dtype=torch.bool)

    active = torch.where(~is_placeholder)[0]

    if active.numel() == 0:
        return SlicedWaveform(
            adc=torch.zeros(0, device=device, dtype=torch.float32),
            offsets=torch.zeros(1, device=device, dtype=torch.long),
            t0_ns=torch.zeros(0, device=device, dtype=torch.float32),
            pmt_id=torch.zeros(0, device=device, dtype=torch.long),
            tick_ns=sw.tick_ns,
            n_channels=sw.n_channels,
            n_bins=sw.n_bins,
            attrs=dict(sw.attrs),
        )

    active_channels = torch.unique(sw.pmt_id[active])
    active_channels_list = active_channels.tolist()
    n_active_ch = active_channels.numel()

    # Per-channel earliest start and latest end, then take global extremes
    ch_t0_list = []
    ch_t_end_list = []
    for ch in active_channels_list:
        ch_active = active[sw.pmt_id[active] == ch]
        ch_t0 = sw.t0_ns[ch_active].min().item()
        ch_t_end = (sw.t0_ns[ch_active] + chunk_lengths[ch_active].float() * sw.tick_ns).max().item()
        ch_t0_list.append(ch_t0)
        ch_t_end_list.append(ch_t_end)

    t0_global = min(ch_t0_list)
    t_end_global = max(ch_t_end_list)
    n_bins_global = max(1, int(round((t_end_global - t0_global) / sw.tick_ns)))

    # Pre-filled with `fill`. left-pad comes from r0 offset, right-pad is implicit
    new_adc = torch.full(
        (n_active_ch * n_bins_global,), fill, device=device, dtype=torch.float32
    )

    for i, ch in enumerate(active_channels_list):
        ch_active = active[sw.pmt_id[active] == ch]
        row = new_adc[i * n_bins_global : (i + 1) * n_bins_global]
        for k in ch_active.tolist():
            chunk_data = sw.adc[sw.offsets[k] : sw.offsets[k + 1]]
            # r0 is offset from the global t0, giving left-padding for late-starting channels
            r0 = int(round((sw.t0_ns[k].item() - t0_global) / sw.tick_ns))
            chunk_len = chunk_data.numel()
            end = min(r0 + chunk_len, n_bins_global)
            if end > r0:
                row[r0:end] = chunk_data[: end - r0]

    new_offsets = torch.arange(
        0, (n_active_ch + 1) * n_bins_global, n_bins_global,
        device=device, dtype=torch.long,
    )
    new_t0_ns = torch.full((n_active_ch,), t0_global, device=device, dtype=torch.float32)

    return SlicedWaveform(
        adc=new_adc,
        offsets=new_offsets,
        t0_ns=new_t0_ns,
        pmt_id=active_channels.to(dtype=torch.long),
        tick_ns=sw.tick_ns,
        n_channels=sw.n_channels,
        n_bins=n_bins_global,
        attrs=dict(sw.attrs),
    )