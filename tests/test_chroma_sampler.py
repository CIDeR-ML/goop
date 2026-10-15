"""ChromaTOFSampler: photons propagated through a chroma-lar detector by TriChroma."""

import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("chroma_lar") is None,
    reason="needs a CUDA GPU and chroma-lar (with TriChroma)",
)


@pytest.fixture(scope="module")
def sampler():
    from goop.sampler import create_chroma_tof_sampler

    return create_chroma_tof_sampler(seed=3)


def test_hits_map_back_to_their_steps(sampler):
    # one source on each side of the cathode, at different emission times
    pos = torch.tensor([[-1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [-500.0, 300.0, -200.0]], device="cuda")
    n_photons = torch.tensor([2e5, 2e5, 1e5], device="cuda")
    t_step = torch.tensor([0.0, 1000.0, 5000.0], device="cuda")
    times, channels, source_idx = sampler.sample(pos, n_photons, t_step)

    assert sampler.n_channels == 162
    assert times.shape == channels.shape == source_idx.shape and len(times) > 0
    assert int(channels.min()) >= 0 and int(channels.max()) < sampler.n_channels
    assert set(source_idx.unique().tolist()) == {0, 1, 2}
    assert bool((times >= t_step[source_idx]).all())
    # the opaque cathode: x<0 light reaches PMTs 0..80, x>0 light PMTs 81..161
    on_positive_side = (channels >= 81)
    assert bool((on_positive_side == (source_idx == 1)).all())


def test_no_photons_gives_no_hits(sampler):
    times, channels, source_idx = sampler.sample(torch.zeros(2, 3, device="cuda"),
                                                 torch.zeros(2, device="cuda"), None)
    assert len(times) == len(channels) == len(source_idx) == 0
