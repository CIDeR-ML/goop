"""Preflight checks for running GOOP production on a new cluster."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_PREFIXES = ("/path/to/", "PATH/TO/")


def _repo_relative(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _is_placeholder(path: str | None) -> bool:
    if not path:
        return True
    return str(path).startswith(PLACEHOLDER_PREFIXES)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data:
        config.setdefault("run", {})["data"] = args.data
    if args.outdir:
        config.setdefault("output", {})["outdir"] = args.outdir
    if args.detector_config:
        config.setdefault("detector", {})["config"] = args.detector_config
    if args.plib_path:
        config.setdefault("sampler", {})["plib_path"] = args.plib_path
    if args.sampler_device:
        config.setdefault("sampler", {})["device"] = args.sampler_device


def _check_import(module: str, failures: list[str]) -> Any:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        failures.append(f"import {module}: {exc}")
        print(f"[FAIL] import {module}: {exc}")
        return None
    print(f"[ OK ] import {module}")
    return imported


def _check_path(label: str, path: str | None, failures: list[str], *, required: bool = True) -> None:
    if _is_placeholder(path):
        msg = f"{label} is not set to a real path"
        if required:
            failures.append(msg)
            print(f"[FAIL] {msg}")
        else:
            print(f"[WARN] {msg}")
        return

    resolved = _repo_relative(path)
    assert resolved is not None
    if resolved.exists():
        print(f"[ OK ] {label}: {resolved}")
    elif required:
        failures.append(f"{label} does not exist: {resolved}")
        print(f"[FAIL] {label} does not exist: {resolved}")
    else:
        print(f"[WARN] {label} does not exist: {resolved}")


def _check_output_dir(path: str | None, failures: list[str]) -> None:
    if _is_placeholder(path):
        failures.append("output.outdir is not set to a real path")
        print("[FAIL] output.outdir is not set to a real path")
        return
    resolved = _repo_relative(path)
    assert resolved is not None
    parent = resolved if resolved.exists() else resolved.parent
    if parent.exists() and os.access(parent, os.W_OK):
        print(f"[ OK ] output parent is writable: {parent}")
        return
    failures.append(f"output parent is not writable or does not exist: {parent}")
    print(f"[FAIL] output parent is not writable or does not exist: {parent}")


def _check_gpu(torch_mod: Any, jax_mod: Any, failures: list[str], *, skip_gpu: bool) -> None:
    if skip_gpu:
        print("[SKIP] GPU checks")
        return
    if torch_mod is not None:
        try:
            if torch_mod.cuda.is_available():
                print(f"[ OK ] torch CUDA: {torch_mod.cuda.get_device_name(0)}")
            else:
                failures.append("torch CUDA is not available")
                print("[FAIL] torch CUDA is not available")
        except Exception as exc:
            failures.append(f"torch CUDA check failed: {exc}")
            print(f"[FAIL] torch CUDA check failed: {exc}")
    if jax_mod is not None:
        try:
            devices = jax_mod.devices()
            gpu_devices = [d for d in devices if str(getattr(d, "platform", "")).lower() == "gpu"]
            if gpu_devices:
                print(f"[ OK ] jax GPU devices: {gpu_devices}")
            else:
                failures.append(f"jax sees no GPU devices: {devices}")
                print(f"[FAIL] jax sees no GPU devices: {devices}")
        except Exception as exc:
            failures.append(f"jax device check failed: {exc}")
            print(f"[FAIL] jax device check failed: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether this clone is ready to run GOOP production.")
    parser.add_argument("--run-config", default="production/configs/portable_template.yml")
    parser.add_argument("--data", help="Override run.data for the check")
    parser.add_argument("--outdir", help="Override output.outdir for the check")
    parser.add_argument("--detector-config", help="Override detector.config for the check")
    parser.add_argument("--plib-path", help="Override sampler.plib_path for the check")
    parser.add_argument("--sampler-device", help="Override sampler.device for the check")
    parser.add_argument("--skip-assets", action="store_true", help="Do not require data/photon-library assets to exist")
    parser.add_argument("--skip-gpu", action="store_true", help="Do not require Torch/JAX GPU visibility")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures: list[str] = []

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "jaxtpc"))

    print(f"Repository: {REPO_ROOT}")
    print(f"Run config: {args.run_config}")

    np_mod = _check_import("numpy", failures)
    h5py_mod = _check_import("h5py", failures)
    yaml_mod = _check_import("yaml", failures)
    torch_mod = _check_import("torch", failures)
    jax_mod = _check_import("jax", failures)
    _check_import("goop", failures)
    _check_import("tools.loader", failures)
    _ = (np_mod, h5py_mod, yaml_mod)

    cfg_path = _repo_relative(args.run_config)
    if cfg_path is None or not cfg_path.exists():
        failures.append(f"run config does not exist: {cfg_path}")
        print(f"[FAIL] run config does not exist: {cfg_path}")
        config: dict[str, Any] = {}
    else:
        print(f"[ OK ] run config exists: {cfg_path}")
        config = _load_yaml(cfg_path)
        _apply_overrides(config, args)

    run = config.get("run", {})
    detector = config.get("detector", {})
    output = config.get("output", {})
    sampler = config.get("sampler", {})

    asset_required = not args.skip_assets
    _check_path("run.data", run.get("data"), failures, required=asset_required)
    _check_path("detector.config", detector.get("config"), failures, required=True)
    _check_path("sampler.plib_path", sampler.get("plib_path"), failures, required=asset_required)
    if sampler.get("type") == "siren":
        _check_path("sampler.ckpt_path", sampler.get("ckpt_path"), failures, required=asset_required)
        _check_path("sampler.cfg_path", sampler.get("cfg_path"), failures, required=asset_required)
        _check_path("sampler.sirentv_src", sampler.get("sirentv_src"), failures, required=asset_required)
    _check_output_dir(output.get("outdir"), failures)
    _check_gpu(torch_mod, jax_mod, failures, skip_gpu=args.skip_gpu)

    if failures:
        print("\nPreflight failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
