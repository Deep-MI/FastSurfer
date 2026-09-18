"""The image CerebNet measures its statistics on has to sit on the segmentation's grid.

`pv_calc` pairs the two voxel by voxel and only compares their shapes, so a statistics image that
agrees in shape but not in position would be measured against the wrong voxels. Reslicing into the
target rather than conforming both separately makes the correspondence hold by construction.

The intensities are left alone on the way. They are read for partial volume estimates, which depend
on local contrast, and for intensity statistics, which are reported in the image's own units, so
rescaling would only undo the white matter normalisation the bias field correction applied.
"""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from CerebNet.inference import Inference

CONFORM = {"vox_size": 1.0, "img_size": "auto", "orientation": "native", "order": 1, "dtype": np.uint8}


def grid(offset: float = 0.0) -> np.ndarray:
    """A 1mm grid, optionally shifted, since two conform targets differ only in their offset."""
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = -8.0 + offset
    return affine


def norm_on_grid(tmp_path: Path, data: np.ndarray, norm_affine: np.ndarray) -> tuple[Path, np.ndarray]:
    """Run the norm through `_norm_on_segmentation_grid` against a target on `grid()`."""
    norm_file = tmp_path / "norm.mgz"
    nib.save(nib.MGHImage(data, norm_affine), norm_file)
    inference = SimpleNamespace(_conform_kwargs=dict(CONFORM))
    return Inference._norm_on_segmentation_grid(inference, norm_file, nib.MGHImage(data, grid()))


def test_a_norm_already_on_the_grid_keeps_its_values_and_its_filename(tmp_path):
    """Including an offset of float dust, which is what two conformed grids really differ by."""
    data = np.arange(16**3, dtype=np.uint8).reshape(16, 16, 16)

    out_file, out_data = norm_on_grid(tmp_path, data, grid(offset=7.63e-06))

    assert out_file == tmp_path / "norm.mgz"
    assert np.array_equal(out_data, data)
    assert not (tmp_path / "norm.10mm.mgz").exists()


def test_a_norm_on_another_grid_is_resliced_onto_it(tmp_path):
    """A real offset is corrected, and the resliced copy is what the statistics then refer to."""
    data = np.zeros((16, 16, 16), dtype=np.uint8)
    data[4:12, 4:12, 4:12] = 100

    out_file, out_data = norm_on_grid(tmp_path, data, grid(offset=3.0))

    assert out_file == tmp_path / "norm.10mm.mgz"
    assert out_file.exists()
    assert out_data.shape == data.shape
    assert not np.array_equal(out_data, data)
    # the intensities are carried over rather than rescaled to fill 0-255
    assert out_data.max() == data.max()
