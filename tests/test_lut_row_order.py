"""TOFSampler refuses photon libraries whose rows are not one per grid voxel in lookup order."""

import numpy as np
import pytest

from goop.sampler.lut import _check_row_order

NUMVOX, LO, HI = [3, 4, 2], [-90.0, -60.0, 0.0], [0.0, 60.0, 60.0]


def _grid_pos():
    size = (np.array(HI) - np.array(LO)) / np.array(NUMVOX)
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in NUMVOX), indexing="ij")
    ijk = np.stack([ix.ravel(order="F"), iy.ravel(order="F"), iz.ravel(order="F")], 1)  # x fastest
    return np.array(LO) + (ijk + 0.5) * size


def test_grid_order_passes():
    _check_row_order(_grid_pos(), NUMVOX, LO, HI)


def test_missing_voxels_are_refused():
    with pytest.raises(ValueError, match="23 rows for a"):
        _check_row_order(_grid_pos()[:-1], NUMVOX, LO, HI)


def test_rows_out_of_order_are_refused():
    pos = _grid_pos()
    pos[[5, 9]] = pos[[9, 5]]
    with pytest.raises(ValueError, match="2 of 24 rows .* first is row 5"):
        _check_row_order(pos, NUMVOX, LO, HI)
