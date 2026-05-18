from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from goop.config import (
    apply_flat_overrides,
    aux_photon_sources_config,
    build_aux_photon_sources,
    build_delay_chain,
    delay_chain_config,
    flatten_config_for_argparse,
    normalize_run_config,
    sampler_config,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_nested_component_kwargs_are_siblings():
    cfg = normalize_run_config({
        "sampler": {
            "type": "lut",
            "plib_path": "/tmp/wire.h5",
            "pmt_qe": 0.2,
            "n_simulated": 123,
            "lazy": True,
            "interpolate": False,
            "device": "cpu",
        },
        "delays": {
            "chain": [
                {"type": "TTSDelay", "fwhm_ns": 3.0, "apply_transit_time": True},
            ],
        },
        "aux_photon_sources": [
            {"type": "DarkNoise", "rate_hz": 42.0},
        ],
    })

    sampler = sampler_config(cfg)
    assert sampler["type"] == "lut"
    assert sampler["plib_path"] == "/tmp/wire.h5"
    assert sampler["pmt_qe"] == 0.2
    assert sampler["n_simulated"] == 123
    assert sampler["lazy"] is True
    assert sampler["interpolate"] is False
    assert sampler["device"] == "cpu"

    delays = build_delay_chain(cfg)
    assert len(delays.delays) == 1
    assert delays.delays[0].fwhm_ns == 3.0
    assert delays.delays[0].apply_transit_time is True

    aux = build_aux_photon_sources(cfg)
    assert len(aux) == 1
    assert aux[0].rate_hz == 42.0


def test_flat_legacy_config_maps_to_nested():
    cfg = normalize_run_config({
        "data": "/tmp/input.h5",
        "detector_config": "jaxtpc/config/cubic_wireplane_config.yaml",
        "dataset": "legacy",
        "outdir": "/tmp/out",
        "events": None,
        "events_per_file": 7,
        "digitize": False,
        "dark_noise": True,
        "dark_noise_rate": 1234.0,
        "plib_path": "/tmp/plib.h5",
        "lut_n_simulated": 30_000_000,
        "voxel_dx": 10.0,
    })
    flat = flatten_config_for_argparse(cfg)

    assert flat["data"] == "/tmp/input.h5"
    assert flat["config"] == "jaxtpc/config/cubic_wireplane_config.yaml"
    assert flat["dataset"] == "legacy"
    assert flat["events_per_file"] == 7
    assert flat["no_digitize"] is True
    assert flat["dark_noise"] is True
    assert flat["dark_noise_rate"] == 1234.0
    assert flat["pca_lut_path"] == "/tmp/plib.h5"
    assert flat["lut_n_simulated"] == 30_000_000
    assert flat["voxel_dx"] == 10.0


def test_flat_cli_override_does_not_clear_custom_aux_unless_explicit():
    cfg = normalize_run_config({
        "aux_photon_sources": [
            {"type": "DarkNoise", "rate_hz": 111.0},
        ],
    })

    unchanged = apply_flat_overrides(cfg, {"events": 3})
    assert aux_photon_sources_config(unchanged) == [{"type": "DarkNoise", "rate_hz": 111.0}]

    disabled = apply_flat_overrides(cfg, {"dark_noise": False})
    assert aux_photon_sources_config(disabled) == []


def test_delays_can_be_disabled():
    cfg = normalize_run_config({"delays": {"enabled": False}})
    assert len(build_delay_chain(cfg).delays) == 0
    assert delay_chain_config(cfg) == []


def test_component_kwargs_nesting_is_rejected():
    with pytest.raises(ValueError, match="beside 'type'"):
        normalize_run_config({"sampler": {"type": "lut", "kwargs": {"pmt_qe": 0.2}}})


def test_short_delay_alias_is_rejected():
    cfg = normalize_run_config({"delays": {"chain": [{"type": "tts"}]}})
    with pytest.raises(ValueError, match="Unknown delays.chain"):
        build_delay_chain(cfg)


def test_production_presets_load():
    for rel in [
        "production/configs/out_full_prod.yml",
        "production/configs/out_full_prod_pixel.yml",
    ]:
        cfg = normalize_run_config(
            yaml.safe_load((REPO_ROOT / rel).read_text()),
            rel,
        )
        assert sampler_config(cfg)["type"] == "lut"
        assert delay_chain_config(cfg)
        assert aux_photon_sources_config(cfg) == []
