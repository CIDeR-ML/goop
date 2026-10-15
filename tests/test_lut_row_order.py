"""TOFSampler refuses photon libraries whose rows are not one per grid voxel in lookup order."""

import os

import pytest

from goop.sampler.lut import TOFSampler

PIXEL_B01 = "/sdf/data/neutrino/youngsam/compressed_pixel_plib_b01_quantile_log_n50.h5"  # 18,504 voxels missing
WIRE_B05 = "/sdf/data/neutrino/youngsam/compressed_plib_b05_quantile_log_prod_lite_n50.h5"


@pytest.mark.skipif(not os.path.exists(PIXEL_B01), reason="needs the b01 pixel library")
def test_library_with_missing_and_shuffled_rows_is_refused():
    with pytest.raises(ValueError, match="not one per grid voxel"):
        TOFSampler(PIXEL_B01, lazy=True)


@pytest.mark.skipif(not os.path.exists(WIRE_B05), reason="needs the b05 wire library")
def test_library_in_grid_order_loads():
    TOFSampler(WIRE_B05, lazy=True)
