"""
Batch optical simulation: run jaxtpc light generation + GOOP waveform production.

Produces one file type per batch:
    {dataset}_sensor_optical_{NNNN}.h5  — per-PMT SlicedWaveform data (digitized or raw)

See README.md for pipeline details, output schema, and performance benchmarks.

Usage (from project root):
    python3 production/run_batch.py --data events.h5
    python3 production/run_batch.py --data events.h5 --dataset mpvmpr --events 100
    python3 production/run_batch.py --data events.h5 --events 1000 --events-per-file 100
    python3 production/run_batch.py --run-config production/configs/out_full_prod.yml
"""

import argparse
import concurrent.futures
import gc
import json
import os
import queue
import sys
import threading
import time

# Add project root to path so goop/ is importable,
# and jaxtpc/ so its internal `from tools.X` imports resolve
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'jaxtpc'))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.distributions import Poisson, Uniform, HalfNormal

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_event

from goop import OpticalSimulator
from goop.config import (
    apply_flat_overrides,
    aux_photon_sources_config,
    build_optical_config,
    delay_chain_config,
    flatten_config_for_argparse,
    load_run_config,
    pca_lut_path,
    resolve_digitization,
    sampler_config,
)
from goop.io import write_config_light, save_event_light, save_event_light_w_tpc
from goop.utils import voxelize, throw_in_time_window


sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# HELPERS
# =============================================================================

def get_num_events(data_path):
    with h5py.File(data_path, 'r') as f:
        return f['pstep/lar_vol'].shape[0]

def get_num_labels(dist:torch.distributions.Distribution):
    if dist is None:
        return 10
    return round(dist.sample((1,)).item())

LABEL_FIELDS = {
    'volume': None,           # synthetic: filled with volume index
    'interaction': 'interaction_ids',
    'track': 'track_ids',
    'ancestor': 'ancestor_track_ids',
}


def extract_goop_inputs(filled, cfg, label_key='interaction'):
    """Extract concatenated GOOP inputs from jaxtpc process_event_light output.

    Parameters
    ----------
    label_key : str
        Which field to use as the per-deposit label for GOOP.
        One of 'volume', 'interaction', 'track', 'ancestor'.

    Returns (pos_mm, n_photons, t_step_ns, labels) as JAX arrays,
    plus total_segs count.
    """
    field = LABEL_FIELDS[label_key]
    all_pos, all_nph, all_t, all_labels = [], [], [], []
    all_pdgs, all_des = [], []
    total_segs = 0
    for v in range(cfg.n_volumes):
        vol = filled.volumes[v]
        vol_cfg = cfg.volumes[v]                      # ← config, has geometry
        n = vol.n_actual
        if n == 0:
            continue
        # shift the positions to the global coordinate system of the volume
        pos = vol.positions_mm[:n]
        x_anode_mm  = vol_cfg.x_anode_cm * 10.0
        drift_dir   = vol_cfg.drift_direction
        y_center_mm = vol_cfg.yz_center_cm[0] * 10.0
        z_center_mm = vol_cfg.yz_center_cm[1] * 10.0
        
        x_global = x_anode_mm - drift_dir * pos[:, 0]
        y_global = pos[:, 1] + y_center_mm
        z_global = pos[:, 2] + z_center_mm
        pos = jnp.stack([x_global, y_global, z_global], axis=1)
        total_segs += n
        all_pos.append(pos)
        all_nph.append(jnp.ceil(vol.photons[:n]).astype(jnp.int32))
        all_t.append(vol.t0_us[:n] * 1000.0)  # us -> ns
        all_pdgs.append(vol.pdg[:n])
        all_des.append(vol.de[:n])
        if field is None:
            all_labels.append(jnp.full((n,), v, dtype=jnp.int32))
        else:
            raw = getattr(vol, field)[:n].astype(jnp.int32)
            # Combine volume index into the ID so interactions with the same
            # local ID in different volumes don't collide:
            #   combined = 1_000_000 + 1000 * v + orig_id  (sentinel -1 kept)
            #combined = jnp.where(raw == -1, jnp.int32(-1),
            #                     jnp.int32(1_000_000 + 1000 * v) + raw)
            #all_labels.append(combined)
            all_labels.append(raw)

    pos_mm = jnp.concatenate(all_pos)
    n_photons = jnp.concatenate(all_nph)
    t_step_ns = jnp.concatenate(all_t)
    labels = jnp.concatenate(all_labels)
    pdgs = jnp.concatenate(all_pdgs)
    dEs = jnp.concatenate(all_des)
    return pos_mm, n_photons, t_step_ns, labels, pdgs, dEs, total_segs


def voxelize_labeled(pos_mm, n_photons, t_step_ns, labels, dx,
                     device='cuda'):
    """Voxelize each label's segments independently and concatenate.

    Per-label voxelization preserves the label semantics that drive the
    OpticalSimulator's per-waveform split — segments from different labels
    cannot be merged into the same voxel.

    JAX inputs (from ``extract_goop_inputs``) are zero-copied onto the GPU
    via dlpack and the entire computation runs on `device`. Returns torch
    tensors on `device`, ready to feed into ``simulate``.
    """
    pos = torch.from_dlpack(pos_mm).to(device=device, dtype=torch.float32)
    nph = torch.from_dlpack(n_photons).to(device=device, dtype=torch.long)
    tns = torch.from_dlpack(t_step_ns).to(device=device, dtype=torch.float32)
    lbl = torch.from_dlpack(labels).to(device=device, dtype=torch.long)

    pv_all, npv_all, tv_all, lbl_all = [], [], [], []
    for unique_lbl in torch.unique(lbl).tolist():
        mask = lbl == unique_lbl
        if not bool(mask.any()):
            continue
        p_v, n_v, t_v = voxelize(pos[mask], nph[mask], tns[mask], lbl[mask], dx=dx)
        pv_all.append(p_v)
        npv_all.append(n_v)
        tv_all.append(t_v)
        lbl_all.append(torch.full(
            (p_v.shape[0],), unique_lbl, device=device, dtype=torch.long,
        ))

    return (torch.cat(pv_all),
            torch.cat(npv_all),
            torch.cat(tv_all),
            torch.cat(lbl_all))


def _as_numpy(x):
    """Convert tensor/array-like values to host numpy arrays for HDF5 writes."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description='Batch optical simulation (jaxtpc -> GOOP)')
    parser.add_argument('--run-config', '--run-yml', '--run-yaml',
                        default=None,
                        help='YAML file with defaults for this run. CLI flags '
                             'override values from the file.')
    # I/O
    parser.add_argument('--data', default=defaults['data'], help='Input HDF5 file')
    parser.add_argument('--config', default=defaults['config'],
                        help='Detector geometry YAML')
    parser.add_argument('--dataset', default=defaults['dataset'],
                        help='Dataset name prefix for output files')
    parser.add_argument('--outdir', default=defaults['outdir'], help='Output directory')
    parser.add_argument('--events', type=int, default=defaults['events'],
                        help='Number of events (default: all)')
    parser.add_argument('--events-per-file', type=int,
                        default=defaults['events_per_file'],
                        help='Events per output file (default: 1000)')
    parser.add_argument('--label-key', default=defaults['label_key'],
                        choices=list(LABEL_FIELDS.keys()),
                        help='Deposit label for per-waveform separation '
                             '(default: interaction)')
    parser.add_argument('--label-dist', default=defaults['label_dist'],
                        choices=['Uniform', 'Poisson', 'HalfNormal', 'Fixed'],
                        help='Distribution for label sampling (default: Uniform)')
    # Digitization
    parser.add_argument('--n-bits', type=int, default=defaults['n_bits'],
                        help='ADC bit depth (default: 15)')
    parser.add_argument('--pedestal', type=float, default=defaults['pedestal'],
                        help='ADC pedestal (default: 0.9 * (2^n_bits - 1))')
    parser.add_argument('--max-pe-per-pmt', type=float,
                        default=defaults['max_pe_per_pmt'],
                        help='PE scale for gain calculation (default: 90000)')
    parser.add_argument('--no-digitize', dest='no_digitize',
                        action='store_true', default=defaults['no_digitize'],
                        help='Disable ADC digitization')
    parser.add_argument('--digitize', dest='no_digitize',
                        action='store_false',
                        help='Enable ADC digitization, overriding a YAML no_digitize setting')
    # Physics
    parser.add_argument('--dark-noise', dest='dark_noise',
                        action='store_true', default=defaults['dark_noise'],
                        help='Enable dark noise')
    parser.add_argument('--no-dark-noise', dest='dark_noise',
                        action='store_false',
                        help='Disable dark noise, overriding a YAML dark_noise setting')
    parser.add_argument('--dark-noise-rate', type=float,
                        default=defaults['dark_noise_rate'],
                        help='Dark noise rate in Hz (default: 2000)')
    parser.add_argument('--baseline-noise-std', type=float,
                        default=defaults['baseline_noise_std'],
                        help='Gaussian baseline noise std (default: 0.0)')
    parser.add_argument('--ser-jitter-std', type=float,
                        default=defaults['ser_jitter_std'],
                        help='SER jitter std (default: 0.1)')
    parser.add_argument('--time-window-ns', type=int,
                        default=defaults['time_window_ns'],
                        help='Time window for GOOP simulation in ns (default: 10000)')

    # Timing / oversampling
    parser.add_argument('--tick-ns', type=float, default=defaults['tick_ns'],
                        help='Output time bin width in ns (default: 1.0)')
    parser.add_argument('--oversample', type=int, default=defaults['oversample'],
                        help='Internal oversampling factor (default: 10)')
    # jaxtpc
    parser.add_argument('--total-pad', type=int, default=defaults['total_pad'],
                        help='Padding for segment arrays (default: 250000)')
    parser.add_argument('--response-chunk-size', type=int,
                        default=defaults['response_chunk_size'],
                        help='jaxtpc response chunk size (default: 50000)')
    # Sampler
    parser.add_argument('--sampler', choices=['lut', 'siren'],
                        default=defaults['sampler'],
                        help='TOF sampler backend: lut (voxel LUT, eager) or '
                             'siren (SIREN neural network); default: lut')
    parser.add_argument('--pca-lut-path', '--plib-path',
                        default=defaults['pca_lut_path'],
                        help='PCA photon-library HDF5 path used by the LUT '
                             'sampler, and as the PCA basis for the SIREN sampler')
    parser.add_argument('--pmt-qe', type=float, default=defaults.get('pmt_qe'),
                        help='Sampler PMT quantum efficiency passed to the TOF sampler')
    parser.add_argument('--lut-n-simulated', type=float,
                        default=defaults.get('lut_n_simulated'),
                        help='Number of simulated photons represented by the PCA LUT')
    parser.add_argument('--sampler-device', default=defaults.get('sampler_device'),
                        help='Device string passed to the TOF sampler')
    parser.add_argument('--sampler-lazy', dest='sampler_lazy',
                        action='store_true', default=defaults.get('sampler_lazy'),
                        help='Use lazy HDF5-backed LUT access instead of eager GPU load')
    parser.add_argument('--no-sampler-lazy', dest='sampler_lazy',
                        action='store_false',
                        help='Use eager LUT loading, overriding YAML sampler.lazy')
    parser.add_argument('--sampler-interpolate', dest='sampler_interpolate',
                        action='store_true', default=defaults.get('sampler_interpolate'),
                        help='Enable sampler interpolation')
    parser.add_argument('--no-sampler-interpolate', dest='sampler_interpolate',
                        action='store_false',
                        help='Disable sampler interpolation, overriding YAML sampler.interpolate')
    parser.add_argument('--siren-ckpt-path', default=defaults.get('siren_ckpt_path'),
                        help='SIREN checkpoint path passed to the SIREN sampler')
    parser.add_argument('--siren-cfg-path', default=defaults.get('siren_cfg_path'),
                        help='SIREN train config path passed to the SIREN sampler')
    parser.add_argument('--sirentv-src', default=defaults.get('sirentv_src'),
                        help='sirentv source tree path passed to the SIREN sampler')
    # Voxelization
    parser.add_argument('--voxel-dx', type=float, default=defaults['voxel_dx'],
                        help='Voxelize input segments to a cubic grid of side '
                             'length dx (mm) before goop simulate. Voxelization '
                             'is performed per-label group so per-waveform label '
                             'separation is preserved. 0 disables (default).')
    # Other
    parser.add_argument('--workers', type=int, default=defaults['workers'],
                        help='Number of save worker threads (0=serial, default: 2)')
    parser.add_argument('--seed', type=int, default=defaults['seed'])
    parser.add_argument('--align', dest='align',
                        action='store_true', default=defaults['align'],
                        help='Requires subtract_t0=True in goop_sim.simulate()')
    parser.add_argument('--no-align', dest='align',
                        action='store_false',
                        help='Disable waveform alignment, overriding a YAML align setting')
    return parser


CLI_OVERRIDE_SPECS = [
    ('data', 'data', ['--data']),
    ('config', 'config', ['--config']),
    ('dataset', 'dataset', ['--dataset']),
    ('outdir', 'outdir', ['--outdir']),
    ('events', 'events', ['--events']),
    ('events_per_file', 'events_per_file', ['--events-per-file']),
    ('label_key', 'label_key', ['--label-key']),
    ('label_dist', 'label_dist', ['--label-dist']),
    ('n_bits', 'n_bits', ['--n-bits']),
    ('pedestal', 'pedestal', ['--pedestal']),
    ('max_pe_per_pmt', 'max_pe_per_pmt', ['--max-pe-per-pmt']),
    ('dark_noise_rate', 'dark_noise_rate', ['--dark-noise-rate']),
    ('baseline_noise_std', 'baseline_noise_std', ['--baseline-noise-std']),
    ('ser_jitter_std', 'ser_jitter_std', ['--ser-jitter-std']),
    ('time_window_ns', 'time_window_ns', ['--time-window-ns']),
    ('tick_ns', 'tick_ns', ['--tick-ns']),
    ('oversample', 'oversample', ['--oversample']),
    ('total_pad', 'total_pad', ['--total-pad']),
    ('response_chunk_size', 'response_chunk_size', ['--response-chunk-size']),
    ('sampler', 'sampler', ['--sampler']),
    ('pca_lut_path', 'pca_lut_path', ['--pca-lut-path', '--plib-path']),
    ('pmt_qe', 'pmt_qe', ['--pmt-qe']),
    ('lut_n_simulated', 'lut_n_simulated', ['--lut-n-simulated']),
    ('sampler_device', 'sampler_device', ['--sampler-device']),
    ('siren_ckpt_path', 'siren_ckpt_path', ['--siren-ckpt-path']),
    ('siren_cfg_path', 'siren_cfg_path', ['--siren-cfg-path']),
    ('sirentv_src', 'sirentv_src', ['--sirentv-src']),
    ('workers', 'workers', ['--workers']),
    ('seed', 'seed', ['--seed']),
]


def _option_present(argv, options):
    for token in argv:
        if not token.startswith('--'):
            continue
        opt = token.split('=', 1)[0]
        if opt in options:
            return True
    return False


def _collect_cli_overrides(args, argv):
    overrides = {}
    for attr, key, options in CLI_OVERRIDE_SPECS:
        if _option_present(argv, options):
            overrides[key] = getattr(args, attr)

    if _option_present(argv, ['--no-digitize']):
        overrides['no_digitize'] = True
    elif _option_present(argv, ['--digitize']):
        overrides['digitize'] = True

    if _option_present(argv, ['--dark-noise', '--no-dark-noise']):
        overrides['dark_noise'] = args.dark_noise
        if args.dark_noise:
            overrides['dark_noise_rate'] = args.dark_noise_rate

    if _option_present(argv, ['--sampler-lazy', '--no-sampler-lazy']):
        overrides['sampler_lazy'] = args.sampler_lazy
    if _option_present(argv, ['--sampler-interpolate', '--no-sampler-interpolate']):
        overrides['sampler_interpolate'] = args.sampler_interpolate

    if _option_present(argv, ['--align', '--no-align']):
        overrides['align'] = args.align

    return overrides


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--run-config', '--run-yml', '--run-yaml',
                               dest='run_config')
    known, _ = config_parser.parse_known_args(argv)

    run_config = load_run_config(known.run_config)
    parser_defaults = flatten_config_for_argparse(run_config)

    parser = build_parser(parser_defaults)
    args = parser.parse_args(argv)
    args.run_config = known.run_config

    overrides = _collect_cli_overrides(args, argv)
    args.resolved_config = apply_flat_overrides(run_config, overrides)
    for key, value in flatten_config_for_argparse(args.resolved_config).items():
        setattr(args, key, value)
    return args


# =============================================================================
# MAIN
# =============================================================================

def main(argv=None):
    args = parse_args(argv)
    run_config = args.resolved_config

    digitization = resolve_digitization(run_config)
    include_digitize = digitization['enabled']
    n_bits = digitization['n_bits']
    pedestal = digitization['pedestal']
    gain = digitization['gain']
    events_per_file = args.events_per_file
    dataset_name = args.dataset
    label_key = args.label_key
    label_dist = args.label_dist
    should_align = args.align
    time_window_ns = args.time_window_ns

    total_events = get_num_events(args.data)
    num_events = min(args.events, total_events) if args.events else total_events
    num_files = (num_events + events_per_file - 1) // events_per_file


    if label_dist == 'Uniform':
        label_dist = Uniform(low=1, high=15)
    elif label_dist == 'Poisson':
        label_dist = Poisson(lam=10)
    elif label_dist == 'HalfNormal':
        label_dist = HalfNormal(scale=10)
    elif label_dist == 'Fixed':
        label_dist = None
    else:
        raise ValueError(f'Invalid label distribution: {label_dist}')
    
    N_labels = get_num_labels(label_dist)

    # Output directory
    optical_dir = os.path.join(args.outdir, 'sensor_optical')
    os.makedirs(optical_dir, exist_ok=True)

    print('=' * 60)
    print(' GOOP Batch Optical Simulation')
    print('=' * 60)
    if args.run_config:
        print(f'  Run config:     {args.run_config}')
    print(f'  Data:           {args.data} ({num_events}/{total_events} events)')
    print(f'  Dataset:        {dataset_name}')
    print(f'  Label key:      {label_key}')
    print(f'  Events/file:    {events_per_file}')
    print(f'  Num files:      {num_files}')
    print(f'  Digitization:   {"ON" if include_digitize else "OFF"}')
    if include_digitize:
        print(f'    n_bits:       {n_bits}')
        print(f'    pedestal:     {pedestal:.0f}')
        print(f'    gain:         {gain:.6f}')
    sampler_meta = sampler_config(run_config)
    delay_meta = delay_chain_config(run_config)
    aux_meta = aux_photon_sources_config(run_config)
    print(f'  Sampler:        {sampler_meta.get("type", args.sampler).upper()}')
    print(f'  Sampler config: {json.dumps(sampler_meta, sort_keys=True)}')
    print(f'  Delays:         {json.dumps(delay_meta, sort_keys=True)}')
    print(f'  Aux sources:    {json.dumps(aux_meta, sort_keys=True)}')
    print(f'  Baseline noise: {args.baseline_noise_std}')
    print(f'  Tick:           {args.tick_ns} ns')
    print(f'  Oversample:     {args.oversample}x')
    print(f'  Voxel dx:       {args.voxel_dx} mm'
          + ('  (disabled)' if args.voxel_dx <= 0 else ''))
    print(f'  Total pad:      {args.total_pad:,}')
    print(f'  Workers:        {args.workers} {"(serial)" if args.workers == 0 else "(threaded)"}')
    print(f'  JAX device:     {jax.devices()[0]}')
    print(f'  CUDA device:    {torch.cuda.get_device_name(0)}')
    print(f'  Output:         {optical_dir}/')
    print(f'  Label dist:     {label_dist}')
    print(f'  Time window:    {time_window_ns} ns')
    print()

    # ---- Create jaxtpc simulator (light-only) ----
    t_create = time.time()
    detector_config = generate_detector(args.config)
    jaxtpc_sim = DetectorSimulator(
        detector_config,
        total_pad=args.total_pad,
        response_chunk_size=args.response_chunk_size,
        include_track_hits=False,
    )
    cfg = jaxtpc_sim.config
    t_create = time.time() - t_create
    print(f'  jaxtpc creation:  {t_create:.1f}s')

    # ---- Create GOOP simulator ----
    t_goop = time.time()
    goop_config = build_optical_config(run_config, gain=gain, n_labels=N_labels)
    goop_sim = OpticalSimulator(goop_config)
    t_goop = time.time() - t_goop
    print(f'  GOOP creation:    {t_goop:.1f}s')

    # ---- Warmup ----
    print('  Warmup...', end='', flush=True)
    t_warmup = time.time()
    warmup_dep = load_event(args.data, cfg, event_idx=0)
    warmup_filled = jaxtpc_sim.process_event_light(warmup_dep)
    jax.block_until_ready(warmup_filled.volumes[0].charge)
    pos, nph, tns, lbl, pdg, des, _ = extract_goop_inputs(warmup_filled, cfg, label_key)
    if args.time_window_ns > 0:
        rand_time_tpcs_ns = throw_in_time_window(pos, nph, tns, lbl, time_window_ns=args.time_window_ns, pdgs=pdg, de=des)
        pos = rand_time_tpcs_ns["pos_mm"]
        nph = rand_time_tpcs_ns["n_photons"]
        tns = rand_time_tpcs_ns["t_step"]
        lbl = rand_time_tpcs_ns["labels"]
        pdg = rand_time_tpcs_ns["pdgs"]
        des = rand_time_tpcs_ns["de"]
    if args.voxel_dx > 0:
        pos_vox, nph_vox, tns_vox, lbl_vox = voxelize_labeled(pos, nph, tns, lbl, args.voxel_dx)
        warmup_wfs = goop_sim.simulate(
            pos_vox, nph_vox, tns_vox, labels=lbl_vox, stitched=True, subtract_t0=False if args.time_window_ns > 0 else True
        )
        del warmup_dep, warmup_filled, warmup_wfs, pos, pos_vox, nph, nph_vox, tns, tns_vox, lbl, lbl_vox, pdg, des
    else:
        warmup_wfs = goop_sim.simulate(
            pos, nph, tns, labels=lbl, stitched=True, subtract_t0=False if args.time_window_ns > 0 else True
        )        
        del warmup_dep, warmup_filled, warmup_wfs, pos, nph, tns, lbl, pdg, des
    gc.collect()
    torch.cuda.empty_cache()
    t_warmup = time.time() - t_warmup
    print(f' {t_warmup:.1f}s\n')

    # ---- Save helpers ----
    num_workers = args.workers
    file_lock = threading.Lock()

    def save_one_event(f, item):
        """Save a single event to HDF5. Thread-safe via file_lock."""
        event_key, waveforms_cpu, source_idx = item
        with file_lock:
            save_event_light(
                f, event_key, waveforms_cpu,
                source_event_idx=source_idx,
                digitized=include_digitize,
                n_bits=n_bits if include_digitize else 0)

    def save_one_event_with_tpc(f, item):
        """Save a single event to HDF5 with TPC data. Thread-safe via file_lock."""
        event_key, waveforms_cpu, source_idx, positions, n_photons, t_step, labels, de, pdg = item
        with file_lock:
            save_event_light_w_tpc(
                f, event_key, waveforms_cpu,
                positions=positions,
                n_photons=n_photons,
                t_step=t_step,
                labels=labels,
                de=de,
                pdg=pdg,
                source_event_idx=source_idx,
                digitized=include_digitize,
                n_bits=n_bits if include_digitize else 0)

    def save_worker(f, q):
        """Worker thread: pull items from queue, save to HDF5."""
        while True:
            item = q.get()
            if item is None:
                break
            #save_one_event(f, item)
            save_one_event_with_tpc(f, item)
            q.task_done()

    def waveforms_to_cpu(waveforms):
        """Transfer all waveform tensors to CPU, freeing GPU memory."""
        for wf in waveforms:
            wf.adc = wf.adc.cpu()
            wf.offsets = wf.offsets.cpu()
            wf.t0_ns = wf.t0_ns.cpu()
            wf.pmt_id = wf.pmt_id.cpu()
            pe = wf.attrs.get('pe_counts')
            if pe is not None and isinstance(pe, torch.Tensor):
                wf.attrs['pe_counts'] = pe.cpu()
        return waveforms

    # ---- Process events ----
    total_start = time.time()

    for file_idx in range(num_files):
        event_start = file_idx * events_per_file
        event_end = min(event_start + events_per_file, num_events)
        n_in_file = event_end - event_start

        optical_path = os.path.join(
            optical_dir, f'{dataset_name}_sensor_optical_{file_idx:04d}.h5')

        print(f'File {file_idx:04d}: events {event_start}\u2013{event_end-1} '
              f'({n_in_file} events)')

        with h5py.File(optical_path, 'w') as f:
            write_config_light(
                f, goop_config,
                label_key=label_key,
                n_labels=cfg.n_volumes if label_key == 'volume' else 0,
                dataset_name=dataset_name,
                file_index=file_idx,
                source_file=args.data,
                pca_lut_path=pca_lut_path(run_config),
                sampler_type=sampler_meta.get('type', ''),
                sampler_config=sampler_meta,
                delay_chain=delay_meta,
                aux_photon_sources=aux_meta,
                n_events=n_in_file,
                global_event_offset=event_start,
            )

            # Start workers
            save_queue = None
            workers = []
            if num_workers > 0:
                save_queue = queue.Queue(maxsize=num_workers + 2)
                for _ in range(num_workers):
                    t = threading.Thread(target=save_worker,
                                         args=(f, save_queue), daemon=True)
                    t.start()
                    workers.append(t)

            def prefetch_load(evt_idx):
                """Load event from HDF5 (CPU-only, no GPU work)."""
                return load_event(args.data, cfg, event_idx=evt_idx)

            print(f"  {'Evt':>4} {'Segs':>16} {'Photons':>12} "
                  f"{'Labels':>6} {'PEs':>10} {'Chunks':>7} "
                  f"{'t_load':>6} {'t_light':>7} {'t_vox':>6} "
                  f"{'t_goop':>6} {'t_save':>6} {'total':>6}")
            print(f"  {'-' * 110}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch:
                # Submit first load immediately
                future = prefetch.submit(prefetch_load, event_start)

                for evt_idx in range(event_start, event_end):
                    local_idx = evt_idx - event_start
                    event_key = f'event_{local_idx:03d}'

                    n_labels = get_num_labels(label_dist)
                    goop_config.n_labels_to_simulate = n_labels
                    goop_sim = OpticalSimulator(goop_config)
                    print(f"  sampled {n_labels} interactions from {label_dist}")
                    # 1. Collect prefetched deposits
                    t0 = time.time()
                    deposits = future.result()
                    t_load = time.time() - t0

                    # 2. Submit prefetch for next event
                    if evt_idx + 1 < event_end:
                        future = prefetch.submit(prefetch_load, evt_idx + 1)

                    # 3. jaxtpc light generation (GPU — kept in main thread)
                    t0 = time.time()
                    filled = jaxtpc_sim.process_event_light(deposits)
                    jax.block_until_ready(filled.volumes[0].charge)
                    t_light = time.time() - t0

                    # 4. Extract inputs & run GOOP
                    t0 = time.time()

                    pos_mm, n_ph, t_ns, labels, pdgs, des, total_segs = extract_goop_inputs(
                        filled, cfg, label_key)

                    n_after = total_segs
                    if args.time_window_ns > 0:
                        rand_time_tpcs_ns = throw_in_time_window(pos_mm, n_ph, t_ns, labels, time_window_ns=args.time_window_ns, pdgs=pdgs, de=des)
                        pos_mm = rand_time_tpcs_ns["pos_mm"]
                        n_ph = rand_time_tpcs_ns["n_photons"]
                        t_ns = rand_time_tpcs_ns["t_step"]
                        labels = rand_time_tpcs_ns["labels"]
                        pdgs = rand_time_tpcs_ns["pdgs"]
                        des = rand_time_tpcs_ns["de"]
                        n_after = pos_mm.shape[0]

                    # Snapshot the post-window, pre-voxelization per-segment
                    # arrays as numpy. These are the TPC inputs saved alongside
                    # the simulated waveforms; voxelization is only a simulator
                    # acceleration path.
                    raw_pos = _as_numpy(pos_mm)
                    raw_nph = _as_numpy(n_ph)
                    raw_tns = _as_numpy(t_ns)
                    raw_lbl = _as_numpy(labels)
                    raw_pdg = _as_numpy(pdgs)
                    raw_des = _as_numpy(des)

                    sim_pos, sim_nph, sim_tns, sim_labels = pos_mm, n_ph, t_ns, labels
                    t_vox = 0.0
                    if args.voxel_dx > 0:
                        tv0 = time.time()
                        sim_pos, sim_nph, sim_tns, sim_labels = voxelize_labeled(
                            pos_mm, n_ph, t_ns, labels, args.voxel_dx,
                        )
                        t_vox = time.time() - tv0
                        n_after = sim_pos.shape[0]
                    # Re-anchor t0 so t_goop_elapsed measures only goop_sim.simulate.
                    t0 = time.time()
                    waveforms = goop_sim.simulate(
                        sim_pos, sim_nph, sim_tns, labels=sim_labels,
                        stitched=True, subtract_t0=False if args.time_window_ns > 0 else True)

                    # 4.1 - Align Waveforms
                    if should_align:
                        waveforms = [wf.align() for wf in waveforms] # List[SlicedWaveform]
                    
                    t_goop_elapsed = time.time() - t0

                    # Collect stats before CPU transfer
                    n_labels_evt = len(waveforms)
                    total_pe = sum(
                        wvfm.attrs['pe_counts'].sum().item()
                        for wvfm in waveforms)
                    total_chunks = sum(wvfm.n_chunks for wvfm in waveforms)
                    total_photons = int(sim_nph.sum())

                    # 4. GPU → CPU transfer + save (serial or queued)
                    t0 = time.time()
                    waveforms_cpu = waveforms_to_cpu(waveforms)
                    item = (event_key, waveforms_cpu, evt_idx,
                            raw_pos, raw_nph, raw_tns, raw_lbl, raw_des, raw_pdg)

                    if num_workers > 0:
                        save_queue.put(item)
                    else:
                        save_one_event_with_tpc(f, item)
                    t_save = time.time() - t0

                    t_total = t_load + t_light + t_vox + t_goop_elapsed + t_save

                    segs_str = (f'{total_segs:,}->{n_after:,}'
                                if args.voxel_dx > 0 else f'{total_segs:,}')
                    print(f"  {evt_idx:>4} {segs_str:>16} "
                          f"{total_photons:>12,} "
                          f"{n_labels_evt:>6} {total_pe:>10,} "
                          f"{total_chunks:>7,} "
                          f"{t_load:>5.2f}s {t_light:>6.2f}s "
                          f"{t_vox:>5.2f}s "
                          f"{t_goop_elapsed:>5.2f}s {t_save:>5.2f}s "
                          f"{t_total:>5.1f}s")

                    del deposits, filled, waveforms, waveforms_cpu
                    del pos_mm, n_ph, t_ns, labels, pdgs, des
                    del sim_pos, sim_nph, sim_tns, sim_labels
                    del raw_pos, raw_nph, raw_tns, raw_lbl, raw_pdg, raw_des
                    gc.collect()

            # Wait for workers to finish
            if num_workers > 0:
                save_queue.join()
                for _ in range(num_workers):
                    save_queue.put(None)
                for t in workers:
                    t.join()

        # File size
        optical_mb = os.path.getsize(optical_path) / (1024 * 1024)
        print(f'  \u2192 optical: {optical_mb:.1f} MB '
              f'({optical_mb / n_in_file * 1024:.1f} KB/event)')
        print()

    total_elapsed = time.time() - total_start
    print(f'{"=" * 60}')
    print(f'  Done. {num_events} events in {total_elapsed:.1f}s')
    print(f'  Average: {total_elapsed/num_events:.2f}s/event')
    print(f'  Files:   {num_files} in {optical_dir}/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
