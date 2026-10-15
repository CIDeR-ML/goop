"""SlicedWaveform.convolve and .downsample give bit for bit the output of the per-chunk code they replace."""

import math

import pytest
import torch
import torch.nn.functional as F

from goop import SlicedWaveform
from goop.waveform_utils import _next_fft_size

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _sliced(device, lengths, pmt_ids, t0s, n_channels=4):
    g = torch.Generator().manual_seed(1)
    adc = torch.rand(sum(lengths), generator=g).to(device)
    offsets = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device=device)
    return SlicedWaveform(adc=adc, offsets=offsets, t0_ns=torch.tensor(t0s, device=device, dtype=torch.float32),
                          pmt_id=torch.tensor(pmt_ids, device=device), tick_ns=0.1, n_channels=n_channels,
                          n_bins=5000)


def _reference_convolve(sw, kernel, gain):
    """The per-chunk loop of goop 5a3d7ab."""
    conv_ticks = kernel.shape[0]
    max_t0 = torch.full((sw.n_channels,), float("-inf"), device=sw.adc.device)
    max_t0 = max_t0.scatter_reduce(0, sw.pmt_id, sw.t0_ns, reduce="amax", include_self=True)
    is_last = (sw.t0_ns == max_t0[sw.pmt_id]).tolist()
    pieces, offsets = [], [0]
    for k in range(sw.n_chunks):
        chunk = sw.adc[sw.offsets[k]:sw.offsets[k + 1]]
        out_len = chunk.numel() + (conv_ticks - 1 if is_last[k] else 0)
        n_fft = _next_fft_size(chunk.numel() + conv_ticks - 1)
        padded = F.pad(chunk, (0, n_fft - chunk.numel()))
        result = gain * torch.fft.irfft(torch.fft.rfft(padded) * torch.fft.rfft(kernel, n=n_fft), n=n_fft)
        pieces.append(result[:out_len])
        offsets.append(offsets[-1] + out_len)
    return torch.cat(pieces), offsets


def _reference_downsample(sw, factor):
    """The per-chunk loop of goop 5a3d7ab."""
    pieces, offsets = [], [0]
    for k in range(sw.n_chunks):
        chunk = sw.adc[sw.offsets[k]:sw.offsets[k + 1]]
        if chunk.numel() % factor:
            chunk = F.pad(chunk, (0, factor - chunk.numel() % factor))
        pieces.append(chunk.reshape(-1, factor).mean(dim=1))
        offsets.append(offsets[-1] + pieces[-1].numel())
    return torch.cat(pieces), offsets


# chunk lengths that land in three FFT sizes, several chunks per PMT, one PMT without chunks
LENGTHS = [3, 700, 1, 64, 1500, 257, 9, 2048, 33]
PMTS = [0, 0, 0, 1, 1, 2, 2, 2, 3]
T0S = [0.0, 500.0, 900.0, 10.0, 400.0, 0.0, 300.0, 600.0, 50.0]


@pytest.mark.parametrize("device", DEVICES)
def test_convolve_matches_per_chunk_reference(device):
    sw = _sliced(device, LENGTHS, PMTS, T0S, n_channels=5)
    kernel = torch.exp(-torch.arange(600, device=device) / 80.0)
    out = sw.convolve(kernel, gain=-45.0)
    adc, offsets = _reference_convolve(sw, kernel, -45.0)
    assert out.offsets.tolist() == offsets
    assert torch.equal(out.adc, adc)
    assert out.n_bins == sw.n_bins + kernel.shape[0] - 1
    assert torch.equal(out.t0_ns, sw.t0_ns) and torch.equal(out.pmt_id, sw.pmt_id)


@pytest.mark.parametrize("device", DEVICES)
def test_convolve_in_small_batches_matches(device, monkeypatch):
    import goop.waveform as waveform
    sw = _sliced(device, LENGTHS, PMTS, T0S, n_channels=5)
    kernel = torch.exp(-torch.arange(600, device=device) / 80.0)
    whole = sw.convolve(kernel, gain=2.0)
    monkeypatch.setattr(waveform, "_FFT_BATCH_ELEMENTS", 1)  # one chunk per FFT batch
    assert torch.equal(sw.convolve(kernel, gain=2.0).adc, whole.adc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("factor", [1, 7, 10])
def test_downsample_matches_per_chunk_reference(device, factor):
    sw = _sliced(device, LENGTHS, PMTS, T0S)
    out = sw.downsample(factor)
    if factor == 1:
        assert out is sw
        return
    adc, offsets = _reference_downsample(sw, factor)
    assert out.offsets.tolist() == offsets
    assert torch.equal(out.adc, adc)
    assert out.n_bins == math.ceil(sw.n_bins / factor) and out.tick_ns == pytest.approx(sw.tick_ns * factor)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_waveform(device):
    sw = _sliced(device, [], [], [])
    kernel = torch.ones(16, device=device)
    conv = sw.convolve(kernel, gain=1.0)
    assert conv.adc.numel() == 0 and conv.offsets.tolist() == [0] and conv.n_bins == 16
    down = sw.downsample(10)
    assert down.adc.numel() == 0 and down.offsets.tolist() == [0]
