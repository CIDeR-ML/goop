"""Optical TPC simulation pipeline package."""

from .base import ConvolutionKernelBase, DelaySamplerBase, PhotonSourceBase, TOFSamplerBase
from .config import (
    build_aux_photon_sources,
    build_delay_chain,
    build_optical_config,
    build_tof_sampler,
    load_run_config,
    normalize_run_config,
)
from .delays import (
    Delays,
    ScintillationBiexponentialDelay,
    TPBExponentialDelay,
    TTSDelay,
    create_default_delays,
)
from .diff_simulator import DifferentiableOpticalSimulator
from .digitize import DigitizationConfig
from .io import load_event_light, save_event_light, write_config_light, save_event_light_w_tpc, load_event_light_w_tpc
from .kernels import (
    Response,
    RLCKernel,
    ScintillationKernel,
    SERKernel,
    TPBExponentialKernel,
    TPBTriexponentialKernel,
    TTSKernel,
    create_default_response,
)
from .noise import DarkNoise
from .sampler import (
    DifferentiableTOFSampler,
    PCATOFSampler,
    SirenTOFSampler,
    TOFSampler,
    create_default_tof_sampler,
    create_siren_tof_sampler,
)
from .simulator import OpticalSimConfig, OpticalSimulator
from .utils import voxelize
from .waveform import SlicedWaveform, Waveform

__all__ = [
    "ConvolutionKernelBase",
    "DelaySamplerBase",
    "PhotonSourceBase",
    "TOFSamplerBase",
    "load_run_config",
    "normalize_run_config",
    "build_tof_sampler",
    "build_delay_chain",
    "build_aux_photon_sources",
    "build_optical_config",
    "OpticalSimConfig",
    "ScintillationBiexponentialDelay",
    "TPBExponentialDelay",
    "TTSDelay",
    "Delays",
    "create_default_delays",
    "DigitizationConfig",
    "DarkNoise",
    "RLCKernel",
    "Response",
    "ScintillationKernel",
    "SERKernel",
    "TPBExponentialKernel",
    "TPBTriexponentialKernel",
    "TTSKernel",
    "create_default_response",
    "DifferentiableOpticalSimulator",
    "DifferentiableTOFSampler",
    "PCATOFSampler",
    "TOFSampler",
    "SirenTOFSampler",
    "create_default_tof_sampler",
    "create_siren_tof_sampler",
    "OpticalSimulator",
    "Waveform",
    "SlicedWaveform",
    "voxelize",
    "write_config_light",
    "save_event_light",
    "load_event_light",
]
