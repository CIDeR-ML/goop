"""Site-dependent default paths (NERSC Perlmutter vs SLAC s3df).

Every absolute path that differs between the two machines lives here and only here, so
the samplers and notebooks stay identical on both. Mirrors the same idea as sirentv's
`scripts/site_env.sh`.

Selection order:
  1. explicit argument, e.g. ``sites.paths(site="nersc")`` or ``set_site("s3df")``
  2. the ``GOOP_SITE`` environment variable
  3. auto-detection: ``NERSC_HOST`` in the environment -> nersc; ``/sdf`` exists -> s3df

Usage from a notebook::

    from goop import sites
    sites.set_site("nersc")           # or leave it to auto-detect
    p = sites.paths()

Every entry can be overridden per call, so a one-off checkpoint needs no edit here::

    create_siren_tof_sampler(ckpt_path="/my/other.ckpt")

The three legs of the comparison, all from the same b05 production so they share one
voxel grid and one decode:

    compressed quantile plib      `quantile_plib`  QuantileTOFSampler  (trilinear LUT)
    PCA-reconstructed plib        `pca_plib`       TOFSampler          (trilinear LUT)
    SIREN-TV trained on quantile  `q3_ckpt_dir`    SirenTOFSampler     (network)
"""

import os

__all__ = ["SITES", "resolve_site", "set_site", "paths"]

SITES = ("nersc", "s3df")

_PATHS = {
    "nersc": {
        "pca_plib": "/pscratch/sd/j/junjiex/SIREN/data/"
                    "compressed_plib_b05_quantile_log_prod_lite_n50.h5",
        "quantile_plib": "/pscratch/sd/j/junjiex/SIREN/data/"
                         "full_wvfm_plib_b05_quantile_prod_lite_fixed.h5",
        "sirentv_src": "/global/cfs/cdirs/m5238/users/junjiex/CIDER-ML/sirentv",
        "q3_ckpt_dir": "/pscratch/sd/j/junjiex/SIREN/logs/"
                       "logs_train_sirentv_81_q3_official/version-00",
        "q3_cfg": "/pscratch/sd/j/junjiex/SIREN/logs/"
                  "logs_train_sirentv_81_q3_official/version-00/train_cfg.yaml",
    },
    "s3df": {
        "pca_plib": "/sdf/data/neutrino/youngsam/"
                    "compressed_plib_b05_quantile_log_prod_lite_n50.h5",
        "quantile_plib": "/sdf/data/neutrino/pubdata/lut/optical/consolidated/"
                         "full_wvfm_plib_b05_quantile_prod_lite_fixed.h5",
        "sirentv_src": "/sdf/data/neutrino/junjie/SIREN/cider-ml/sirentv",
        "q3_ckpt_dir": "/sdf/data/neutrino/junjie/SIREN/cider-ml/logs/"
                       "logs_train_sirentv_81_q3_official/version-00",
        "q3_cfg": "/sdf/data/neutrino/junjie/SIREN/cider-ml/logs/"
                  "logs_train_sirentv_81_q3_official/version-00/train_cfg.yaml",
    },
}

# Set by set_site(); None means "decide from the environment on every call".
_forced_site = None


def resolve_site(site=None):
    """Return "nersc" or "s3df" following the selection order in the module docstring."""
    cand = site or _forced_site or os.environ.get("GOOP_SITE")
    if cand:
        cand = str(cand).lower()
        if cand not in SITES:
            raise ValueError(f"unknown site {cand!r}; expected one of {SITES}")
        return cand
    if os.environ.get("NERSC_HOST"):
        return "nersc"
    if os.path.isdir("/sdf"):
        return "s3df"
    raise RuntimeError(
        "cannot identify the site (no NERSC_HOST, no /sdf). Pass site=... , call "
        "goop.sites.set_site(...), or set GOOP_SITE=nersc|s3df."
    )


def set_site(site):
    """Pin the site for this process. Pass None to go back to auto-detection."""
    global _forced_site
    if site is not None:
        site = str(site).lower()
        if site not in SITES:
            raise ValueError(f"unknown site {site!r}; expected one of {SITES}")
    _forced_site = site
    return _forced_site


def paths(site=None):
    """Return the path dict for `site` (a copy, so callers can mutate it freely)."""
    return dict(_PATHS[resolve_site(site)])
