# GOOP Production Pipeline

Batch optical simulation of particle events in a liquid argon TPC: jaxtpc photon generation followed by GOOP waveform production, producing structured HDF5 output with per-PMT SlicedWaveform data.

## Contents

```
production/
├── run_batch.py         # Main batch simulation script
├── load.py              # HDF5 load/decode functions
└── README.md            # This file
```

## Usage

From the project root:

```bash
# Basic run (5 events, digitization on)
python3 production/run_batch.py --data events.h5 --events 5

# Full options
python3 production/run_batch.py \
    --data mpvmpr_20.h5 \
    --config jaxtpc/config/cubic_wireplane_config.yaml \
    --dataset myrun \
    --outdir output/ \
    --events 1000 \
    --events-per-file 100 \
    --n-bits 15 \
    --oversample 10 \
    --dark-noise \
    --baseline-noise-std 2.6
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--data` | `out.h5` | Input HDF5 file (edep-sim output) |
| `--config` | `jaxtpc/config/cubic_wireplane_config.yaml` | Detector geometry YAML |
| `--dataset` | `sim` | Dataset name prefix for output files |
| `--outdir` | `.` | Output directory (creates `sensor/` subdir) |
| `--events` | all | Number of events to process |
| `--events-per-file` | 1000 | Events per output HDF5 file |
| `--label-key` | `interaction` | Per-waveform label: `interaction`, `track`, `ancestor`, or `volume` |
| `--n-bits` | 15 | ADC bit depth |
| `--pedestal` | `0.9 * (2^n_bits - 1)` | ADC pedestal value |
| `--max-pe-per-pmt` | 90000 | PE scale for gain calculation |
| `--no-digitize` | (digitize ON) | Disable ADC digitization |
| `--dark-noise` | off | Enable dark noise |
| `--dark-noise-rate` | 2000.0 | Dark noise rate in Hz |
| `--baseline-noise-std` | 0.0 | Gaussian baseline noise std |
| `--ser-jitter-std` | 0.1 | SER weight jitter std |
| `--tick-ns` | 1.0 | Output time bin width in ns |
| `--oversample` | 10 | Internal oversampling factor |
| `--total-pad` | 250,000 | Max deposits per side (jaxtpc JIT shape) |
| `--response-chunk-size` | 50,000 | jaxtpc response chunk size |
| `--lazy` | off | Use lazy (disk-backed) photon library loading |
| `--seed` | 42 | Random seed |

## Pipeline

For each event, `run_batch.py` performs:

1. **Load** particle step data from HDF5 via `load_event`
2. **Light generation** via jaxtpc `process_event_light` — yields per-volume photon counts, positions (mm), and deposit times (us)
3. **Extract** GOOP inputs: `positions_mm`, `ceil(photons).int32`, `t0_us * 1000` (-> ns), per-deposit labels (interaction ID by default; configurable via `--label-key`)
4. **Optical simulation** via GOOP `OpticalSimulator.simulate` — produces one `SlicedWaveform` per unique label value:
   - Photon library lookup + TOF sampling (PCA-compressed, inverse-CDF)
   - Stochastic delays: scintillation (bi-exponential), TPB re-emission (tri-exponential), PMT TTS (Gaussian)
   - Optional dark noise injection
   - Histogramming into per-PMT time bins
   - FFT convolution with SER kernel (10 us duration)
   - Optional baseline noise + ADC digitization (15-bit, pedestal offset, saturation clamping)
5. **Save** per-label `SlicedWaveform` to HDF5 via `save_event_light`

### Label Keys

The `--label-key` flag controls how deposits are grouped into separate waveforms:

| Key | Field | Description |
|---|---|---|
| `interaction` | `interaction_ids` | One waveform per interaction vertex (default) |
| `track` | `track_ids` | One waveform per GEANT4 particle track |
| `ancestor` | `ancestor_track_ids` | One waveform per primary shower ancestor |
| `volume` | synthetic | One waveform per detector volume (east/west) |

## Output File Format

One file type per batch, split by `events_per_file`:

```
{dataset}_sensor_{NNNN}.h5   — per-PMT optical waveforms (SlicedWaveform)
```

### Sensor File Schema

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           tick_ns, n_channels, gain, oversample, ser_jitter_std,
           baseline_noise_std, digitized, n_bits, pedestal,
           kernel_type, label_key, n_labels

/event_{NNN}/
    attrs: source_event_idx, n_labels
    label_{id}/                            one per unique label value
        adc         (N,) uint16            ADC samples (if digitized)
                    (N,) float32           raw signal (if not digitized)
        offsets     (K+1,) int64           CSR chunk boundaries
        t0_ns       (K,) float32           real-time origin per chunk (ns)
        pmt_id      (K,) int32             PMT channel index per chunk
        pe_counts   (C,) int32             PE count per PMT channel
```

**Decode:**
```python
# Chunk k spans adc[offsets[k]:offsets[k+1]]
# with time origin t0_ns[k] on PMT channel pmt_id[k]
# Time axis: t0_ns[k] + arange(chunk_len) * tick_ns
```

## Viewing Output

```python
from production.load import get_file_path, build_viz_config, load_event, list_events

path = get_file_path('output/', 'myrun', file_index=0)
config = build_viz_config(path)  # minimal config from HDF5 metadata
events = list_events(path)       # ['event_000', 'event_001', ...]

waveforms = load_event(path, event_idx=0, device='cpu')
# Returns List[SlicedWaveform], one per volume

for wf in waveforms:
    dense = wf.deslice()  # -> Waveform with shape (n_channels, n_bins)
    print(dense.data.shape, wf.attrs['pe_counts'].sum().item(), 'PEs')
```

## Model Sizes

| Component | Size | Notes |
|---|---|---|
| Photon library | 26 GB | PCA-compressed, quantile-log, 50 components. Loaded into GPU VRAM (eager) or disk-backed (lazy) |
| SER kernel | ~10K samples | 10 us duration at 1 ns tick, oversampled 10x internally |
| 162 PMT channels | 81 per volume | x-reflection symmetry maps half-detector library to full coverage |

## Size Reference

Benchmarked on `out.h5` (MPV/MPR events, ~180K deposits/event average), 15-bit digitization, oversample=10, no baseline noise, `--label-key interaction`:

| Metric | Value |
|---|---|
| Per event | ~6 MB |
| Per 1000 events | ~6 GB |
| Typical PE count | ~1.3M PEs/event |
| Typical interactions/event | ~15 |
| Typical chunks | ~2,500/event |

## Performance

Benchmarked on NVIDIA A100-SXM4-40GB with eager photon library loading, `--label-key interaction`, and `--workers 2`.
Timing below is averaged over events 3–7 (after 3 extra warmup events beyond initial JIT warmup):

| Stage | Time/event |
|---|---|
| Load (HDF5 read) | ~0.3s |
| Light generation (jaxtpc) | <0.01s |
| GOOP simulation | ~2.3s |
| Save (queued to workers) | ~0.03s |
| **Total (main thread)** | **~2.6s** |

- Warmup (JIT + photon library load): ~22s one-time cost
- GOOP simulator creation: ~12s (photon library decompression)
- With `--workers 2`, save runs in background threads — main thread proceeds to next event immediately
- With `--workers 0` (serial), save adds ~3s/event
- Sim time scales with photon count: ~1.6s for small events (~180M photons), ~2.6s for large (~400M photons)
