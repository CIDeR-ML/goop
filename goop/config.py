"""Configuration helpers for GOOP production-style runs."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
import yaml

from .delays import (
    Delays,
    ScintillationBiexponentialDelay,
    TPBExponentialDelay,
    TPBTriexponentialDelay,
    TTSDelay,
)
from .digitize import DigitizationConfig
from .kernels import SERKernel
from .noise import DarkNoise
from .sampler import (
    DEFAULT_N_SIMULATED,
    DEFAULT_PLIB_PATH,
    create_default_tof_sampler,
    create_siren_tof_sampler,
)
from .simulator import OpticalSimConfig


DELAY_REGISTRY = {
    "ScintillationBiexponentialDelay": ScintillationBiexponentialDelay,
    "TPBExponentialDelay": TPBExponentialDelay,
    "TPBTriexponentialDelay": TPBTriexponentialDelay,
    "TTSDelay": TTSDelay,
}

AUX_SOURCE_REGISTRY = {
    "DarkNoise": DarkNoise,
}

SAMPLER_REGISTRY = {
    "lut": create_default_tof_sampler,
    "siren": create_siren_tof_sampler,
}

DEFAULT_DELAY_CHAIN = [
    {
        "type": "ScintillationBiexponentialDelay",
        "singlet_fraction": 0.3,
        "tau_singlet_ns": 6.0,
        "tau_triplet_ns": 1300.0,
    },
    {"type": "TPBTriexponentialDelay"},
    {"type": "TTSDelay", "fwhm_ns": 2.4, "apply_transit_time": True},
]

DEFAULT_RUN_CONFIG = {
    "run": {
        "data": "out.h5",
        "events": None,
        "events_per_file": 1000,
        "label_key": "interaction",
        "label_dist": "Uniform",
        "seed": 42,
        "align": False,
    },
    "detector": {
        "config": "jaxtpc/config/cubic_wireplane_config.yaml",
        "total_pad": 250_000,
        "response_chunk_size": 50_000,
    },
    "output": {
        "dataset": "sim",
        "outdir": ".",
        "workers": 2,
    },
    "digitization": {
        "enabled": True,
        "n_bits": 15,
        "pedestal": None,
        "max_pe_per_pmt": 90_000,
    },
    "optical": {
        "device": "cuda",
        "tick_ns": 1.0,
        "oversample": 10,
        "voxel_dx": 0.0,
        "time_window_ns": 10000,
        "ser_jitter_std": 0.1,
        "baseline_noise_std": 0.0,
        "kernel_duration_ns": 10000,
    },
    "sampler": {
        "type": "lut",
        "plib_path": DEFAULT_PLIB_PATH,
        "n_simulated": DEFAULT_N_SIMULATED,
        "lazy": False,
        "device": "cuda:0",
        "interpolate": True,
        "pmt_qe": 0.12,
    },
    "delays": {
        "enabled": True,
        "chain": DEFAULT_DELAY_CHAIN,
    },
    "aux_photon_sources": [],
}

DEFAULT_SAMPLER_CONFIGS = {
    "lut": DEFAULT_RUN_CONFIG["sampler"],
    "siren": {
        "type": "siren",
        "plib_path": DEFAULT_PLIB_PATH,
        "n_simulated": DEFAULT_N_SIMULATED,
        "device": "cuda:0",
        "pmt_qe": 0.12,
    },
}

SECTION_KEYS = set(DEFAULT_RUN_CONFIG)

SECTION_ALLOWED_KEYS = {
    "run": {"data", "events", "events_per_file", "label_key", "label_dist", "seed", "align"},
    "detector": {"config", "total_pad", "response_chunk_size"},
    "output": {"dataset", "outdir", "workers"},
    "digitization": {"enabled", "n_bits", "pedestal", "max_pe_per_pmt"},
    "optical": {
        "device",
        "tick_ns",
        "oversample",
        "voxel_dx",
        "time_window_ns",
        "ser_jitter_std",
        "baseline_noise_std",
        "kernel_duration_ns",
    },
}

LEGACY_KEY_MAP = {
    "data": ("run", "data"),
    "input": ("run", "data"),
    "input_h5": ("run", "data"),
    "events": ("run", "events"),
    "events_per_file": ("run", "events_per_file"),
    "label_key": ("run", "label_key"),
    "label_dist": ("run", "label_dist"),
    "seed": ("run", "seed"),
    "align": ("run", "align"),
    "config": ("detector", "config"),
    "detector_config": ("detector", "config"),
    "geometry_config": ("detector", "config"),
    "total_pad": ("detector", "total_pad"),
    "response_chunk_size": ("detector", "response_chunk_size"),
    "dataset": ("output", "dataset"),
    "run_name": ("output", "dataset"),
    "outdir": ("output", "outdir"),
    "output": ("output", "outdir"),
    "output_dir": ("output", "outdir"),
    "workers": ("output", "workers"),
    "n_bits": ("digitization", "n_bits"),
    "pedestal": ("digitization", "pedestal"),
    "max_pe_per_pmt": ("digitization", "max_pe_per_pmt"),
    "tick_ns": ("optical", "tick_ns"),
    "oversample": ("optical", "oversample"),
    "voxel_dx": ("optical", "voxel_dx"),
    "time_window_ns": ("optical", "time_window_ns"),
    "ser_jitter_std": ("optical", "ser_jitter_std"),
    "baseline_noise_std": ("optical", "baseline_noise_std"),
    "device": ("optical", "device"),
    "kernel_duration_ns": ("optical", "kernel_duration_ns"),
}

SAMPLER_LEGACY_KEY_MAP = {
    "sampler": "type",
    "pca_lut_path": "plib_path",
    "plib_path": "plib_path",
    "lut_path": "plib_path",
    "pca_path": "plib_path",
    "lut_n_simulated": "n_simulated",
    "n_simulated": "n_simulated",
    "pmt_qe": "pmt_qe",
    "sampler_device": "device",
    "sampler_lazy": "lazy",
    "sampler_interpolate": "interpolate",
    "siren_ckpt_path": "ckpt_path",
    "siren_cfg_path": "cfg_path",
    "sirentv_src": "sirentv_src",
}


def _deepcopy_config(value: Any) -> Any:
    return copy.deepcopy(value)


def _normalize_key(key: Any) -> str:
    return str(key).replace("-", "_")


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    return value


def _default_sampler_for_type(sampler_type: str) -> dict[str, Any]:
    return _deepcopy_config(DEFAULT_SAMPLER_CONFIGS.get(sampler_type, {"type": sampler_type}))


def _validate_section_keys(section: str, value: Mapping[str, Any], path: str) -> None:
    allowed = SECTION_ALLOWED_KEYS.get(section)
    if allowed is None:
        return
    unknown = sorted(set(value) - allowed)
    if unknown:
        valid = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unknown key(s) in {path}.{section}: {', '.join(unknown)}. "
            f"Valid keys: {valid}"
        )


def _merge_simple_section(cfg: dict[str, Any], section: str, value: Any, path: str) -> None:
    value = {_normalize_key(k): v for k, v in _require_mapping(value, f"{path}.{section}").items()}
    _validate_section_keys(section, value, path)
    cfg[section].update(value)


def _merge_sampler(cfg: dict[str, Any], value: Any, path: str) -> None:
    if isinstance(value, str):
        cfg["sampler"] = _default_sampler_for_type(value)
        return
    value = {_normalize_key(k): v for k, v in _require_mapping(value, f"{path}.sampler").items()}
    if "kwargs" in value:
        raise ValueError(
            f"{path}.sampler must put constructor kwargs beside 'type', not under 'kwargs'"
        )
    if "type" in value and value["type"] != cfg["sampler"].get("type"):
        cfg["sampler"] = _default_sampler_for_type(value["type"])
    cfg["sampler"].update(value)


def _merge_delays(cfg: dict[str, Any], value: Any, path: str) -> None:
    if value is None:
        return
    value = {_normalize_key(k): v for k, v in _require_mapping(value, f"{path}.delays").items()}
    unknown = sorted(set(value) - {"enabled", "chain"})
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {path}.delays: {', '.join(unknown)}. "
            "Valid keys: chain, enabled"
        )
    cfg["delays"].update(value)


def _merge_aux_sources(cfg: dict[str, Any], value: Any, path: str) -> None:
    if value is None:
        cfg["aux_photon_sources"] = []
        return
    if not isinstance(value, list):
        raise ValueError(f"{path}.aux_photon_sources must be a list")
    cfg["aux_photon_sources"] = value


def _apply_legacy_key(cfg: dict[str, Any], key: str, value: Any, path: str) -> bool:
    key = _normalize_key(key)
    if key == "digitize":
        cfg["digitization"]["enabled"] = bool(value)
        return True
    if key == "no_digitize":
        cfg["digitization"]["enabled"] = not bool(value)
        return True
    if key == "dark_noise":
        if bool(value):
            rate = cfg.get("_legacy_dark_noise_rate", 2000.0)
            cfg["aux_photon_sources"] = [{"type": "DarkNoise", "rate_hz": rate}]
        else:
            cfg["aux_photon_sources"] = []
        return True
    if key == "dark_noise_rate":
        cfg["_legacy_dark_noise_rate"] = value
        for source in cfg["aux_photon_sources"]:
            if isinstance(source, Mapping) and source.get("type") == "DarkNoise":
                source["rate_hz"] = value
        return True
    if key in SAMPLER_LEGACY_KEY_MAP:
        sampler_key = SAMPLER_LEGACY_KEY_MAP[key]
        if sampler_key == "type" and value != cfg["sampler"].get("type"):
            cfg["sampler"] = _default_sampler_for_type(value)
        else:
            cfg["sampler"][sampler_key] = value
        return True
    if key in LEGACY_KEY_MAP:
        section, section_key = LEGACY_KEY_MAP[key]
        cfg[section][section_key] = value
        return True
    return False


def _strip_internal_keys(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg.pop("_legacy_dark_noise_rate", None)
    return cfg


def normalize_run_config(raw_config: Any = None, path: str = "<config>") -> dict[str, Any]:
    """Return a fully populated nested run config.

    The canonical schema is nested, but legacy flat keys are accepted and
    translated into their nested locations.
    """
    cfg = _deepcopy_config(DEFAULT_RUN_CONFIG)
    if raw_config is None:
        return cfg
    raw_config = _require_mapping(raw_config, path)

    consumed: set[str] = set()
    for section in ("run", "detector", "output", "digitization", "optical"):
        if section in raw_config:
            _merge_simple_section(cfg, section, raw_config[section], path)
            consumed.add(section)
    if "sampler" in raw_config:
        _merge_sampler(cfg, raw_config["sampler"], path)
        consumed.add("sampler")
    if "delays" in raw_config:
        _merge_delays(cfg, raw_config["delays"], path)
        consumed.add("delays")
    if "aux_photon_sources" in raw_config:
        _merge_aux_sources(cfg, raw_config["aux_photon_sources"], path)
        consumed.add("aux_photon_sources")

    for raw_key, value in raw_config.items():
        if raw_key in consumed:
            continue
        key = _normalize_key(raw_key)
        if not _apply_legacy_key(cfg, key, value, path):
            valid = ", ".join(sorted(SECTION_KEYS | set(LEGACY_KEY_MAP) | set(SAMPLER_LEGACY_KEY_MAP) | {
                "dark_noise",
                "dark_noise_rate",
                "digitize",
                "no_digitize",
            }))
            raise ValueError(f"Unknown run config key {raw_key!r} in {path}. Valid keys: {valid}")
    return _strip_internal_keys(cfg)


def apply_flat_overrides(
    base_config: Mapping[str, Any], overrides: Mapping[str, Any], path: str = "<cli>"
) -> dict[str, Any]:
    cfg = _deepcopy_config(base_config)
    for raw_key, value in overrides.items():
        if not _apply_legacy_key(cfg, raw_key, value, path):
            raise ValueError(f"Unknown CLI override key {raw_key!r}")
    return _strip_internal_keys(cfg)


def load_run_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return normalize_run_config()
    with open(path, "r") as f:
        return normalize_run_config(yaml.safe_load(f), path)


def _component_kwargs(spec: Mapping[str, Any], where: str) -> dict[str, Any]:
    spec = {_normalize_key(k): v for k, v in spec.items()}
    if "type" not in spec:
        raise ValueError(f"{where} must include a 'type' field")
    if "kwargs" in spec:
        raise ValueError(f"{where} must put constructor kwargs beside 'type', not under 'kwargs'")
    return {k: v for k, v in spec.items() if k != "type"}


def _build_component(
    spec: Mapping[str, Any],
    registry: Mapping[str, Any],
    where: str,
) -> Any:
    spec = _require_mapping(spec, where)
    component_type = spec.get("type")
    if component_type not in registry:
        valid = ", ".join(sorted(registry))
        raise ValueError(f"Unknown {where} type {component_type!r}. Valid types: {valid}")
    return registry[component_type](**_component_kwargs(spec, where))


def build_tof_sampler(config: Mapping[str, Any]):
    sampler_cfg = _require_mapping(config.get("sampler"), "sampler")
    sampler_type = sampler_cfg.get("type")
    if sampler_type not in SAMPLER_REGISTRY:
        valid = ", ".join(sorted(SAMPLER_REGISTRY))
        raise ValueError(f"Unknown sampler type {sampler_type!r}. Valid types: {valid}")
    return SAMPLER_REGISTRY[sampler_type](**_component_kwargs(sampler_cfg, "sampler"))


def build_delay_chain(config: Mapping[str, Any]) -> Delays:
    delays_cfg = _require_mapping(config.get("delays"), "delays")
    if not delays_cfg.get("enabled", True):
        return Delays([])
    chain = delays_cfg.get("chain", DEFAULT_DELAY_CHAIN)
    if chain is None:
        chain = []
    if not isinstance(chain, list):
        raise ValueError("delays.chain must be a list")
    return Delays([
        _build_component(spec, DELAY_REGISTRY, f"delays.chain[{i}]")
        for i, spec in enumerate(chain)
    ])


def build_aux_photon_sources(config: Mapping[str, Any]) -> list[Any]:
    specs = config.get("aux_photon_sources", [])
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError("aux_photon_sources must be a list")
    return [
        _build_component(spec, AUX_SOURCE_REGISTRY, f"aux_photon_sources[{i}]")
        for i, spec in enumerate(specs)
    ]


def resolve_digitization(config: Mapping[str, Any]) -> dict[str, Any]:
    digitization = _require_mapping(config.get("digitization"), "digitization")
    enabled = bool(digitization.get("enabled", True))
    n_bits = int(digitization.get("n_bits", 15))
    pedestal = digitization.get("pedestal")
    if pedestal is None:
        pedestal = 0.9 * ((1 << n_bits) - 1)
    max_pe_per_pmt = float(digitization.get("max_pe_per_pmt", 90_000))
    gain = (2 ** n_bits) / max_pe_per_pmt
    return {
        "enabled": enabled,
        "n_bits": n_bits,
        "pedestal": float(pedestal),
        "max_pe_per_pmt": max_pe_per_pmt,
        "gain": float(gain),
    }


def build_optical_config(config: Mapping[str, Any], *, gain: float, n_labels: int) -> OpticalSimConfig:
    optical = _require_mapping(config.get("optical"), "optical")
    digitization = resolve_digitization(config)
    device = optical.get("device", "cuda")
    return OpticalSimConfig(
        tof_sampler=build_tof_sampler(config),
        delays=build_delay_chain(config),
        tick_ns=float(optical.get("tick_ns", 1.0)),
        kernel=SERKernel(
            device=torch.device(device),
            duration_ns=float(optical.get("kernel_duration_ns", 10000)),
        ),
        gain=gain,
        n_labels_to_simulate=n_labels,
        time_window_ns=optical.get("time_window_ns", None),
        oversample=int(optical.get("oversample", 1)),
        ser_jitter_std=float(optical.get("ser_jitter_std", 0.0)),
        baseline_noise_std=float(optical.get("baseline_noise_std", 0.0)),
        aux_photon_sources=build_aux_photon_sources(config),
        digitization=(
            DigitizationConfig(
                n_bits=digitization["n_bits"],
                pedestal=digitization["pedestal"],
            )
            if digitization["enabled"] else None
        ),
        device=str(device),
    )


def flatten_config_for_argparse(config: Mapping[str, Any]) -> dict[str, Any]:
    run = config["run"]
    detector = config["detector"]
    output = config["output"]
    digitization = config["digitization"]
    optical = config["optical"]
    sampler = config["sampler"]

    dark_noise = False
    dark_noise_rate = 2000.0
    for source in config.get("aux_photon_sources", []):
        if isinstance(source, Mapping) and source.get("type") == "DarkNoise":
            dark_noise = True
            dark_noise_rate = source.get("rate_hz", dark_noise_rate)
            break

    return {
        "data": run["data"],
        "config": detector["config"],
        "dataset": output["dataset"],
        "outdir": output["outdir"],
        "events": run["events"],
        "events_per_file": run["events_per_file"],
        "label_key": run["label_key"],
        "label_dist": run["label_dist"],
        "n_bits": digitization["n_bits"],
        "pedestal": digitization["pedestal"],
        "max_pe_per_pmt": digitization["max_pe_per_pmt"],
        "no_digitize": not bool(digitization.get("enabled", True)),
        "dark_noise": dark_noise,
        "dark_noise_rate": dark_noise_rate,
        "baseline_noise_std": optical["baseline_noise_std"],
        "ser_jitter_std": optical["ser_jitter_std"],
        "time_window_ns": optical["time_window_ns"],
        "tick_ns": optical["tick_ns"],
        "oversample": optical["oversample"],
        "voxel_dx": optical["voxel_dx"],
        "total_pad": detector["total_pad"],
        "response_chunk_size": detector["response_chunk_size"],
        "sampler": sampler["type"],
        "pca_lut_path": sampler.get("plib_path", DEFAULT_PLIB_PATH),
        "pmt_qe": sampler.get("pmt_qe"),
        "lut_n_simulated": sampler.get("n_simulated"),
        "sampler_device": sampler.get("device"),
        "sampler_lazy": sampler.get("lazy"),
        "sampler_interpolate": sampler.get("interpolate"),
        "siren_ckpt_path": sampler.get("ckpt_path"),
        "siren_cfg_path": sampler.get("cfg_path"),
        "sirentv_src": sampler.get("sirentv_src"),
        "workers": output["workers"],
        "seed": run["seed"],
        "align": run["align"],
    }


def sampler_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return _deepcopy_config(config["sampler"])


def delay_chain_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    delays_cfg = _require_mapping(config.get("delays"), "delays")
    if not delays_cfg.get("enabled", True):
        return []
    return _deepcopy_config(delays_cfg.get("chain", DEFAULT_DELAY_CHAIN))


def aux_photon_sources_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _deepcopy_config(config.get("aux_photon_sources", []) or [])


def pca_lut_path(config: Mapping[str, Any]) -> str:
    return str(config.get("sampler", {}).get("plib_path", ""))
