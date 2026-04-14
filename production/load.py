"""Load utilities for GOOP production output files."""

import os
from typing import Dict, List, Optional

import h5py

from goop.io import load_event_light
from goop.waveform import SlicedWaveform


def get_file_path(outdir: str, dataset: str, file_index: int = 0) -> str:
    """Return the sensor file path for a given dataset and file index."""
    return os.path.join(outdir, 'sensor',
                        f'{dataset}_sensor_{file_index:04d}.h5')


def build_viz_config(sensor_path: str) -> Dict:
    """Build a minimal config dict from HDF5 file metadata."""
    with h5py.File(sensor_path, 'r') as f:
        attrs = dict(f['config'].attrs)
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in attrs.items()}


def load_event(
    sensor_path: str,
    event_idx: int = 0,
    device: str = "cpu",
) -> List[SlicedWaveform]:
    """Load one event from a production sensor file.

    Parameters
    ----------
    sensor_path : str
        Path to ``{dataset}_sensor_{NNNN}.h5``.
    event_idx : int
        Local event index within the file (0-based).
    device : str
        Torch device for loaded tensors.

    Returns
    -------
    list[SlicedWaveform]
        One per detector volume (label). Each has ``.attrs['pe_counts']``
        and ``.attrs['label']``.
    """
    event_key = f'event_{event_idx:03d}'
    with h5py.File(sensor_path, 'r') as f:
        return load_event_light(f, event_key, device=device)


def list_events(sensor_path: str) -> List[str]:
    """Return sorted event keys in a sensor file."""
    with h5py.File(sensor_path, 'r') as f:
        return sorted(k for k in f.keys() if k.startswith('event_'))
