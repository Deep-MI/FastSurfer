"""Regression tests for voxel sizes that carry float32 header noise, e.g. 1.0000001 instead of 1.

recon-surf.sh reads the voxel size of the (already conformed) t1 from its header and passes it on
as an explicit ``--vox_size <float>``, so a nominal 1mm image whose header stores 1.0000001 used to
abort the pipeline at the ``0 < vox_size <= 1`` cap. These tests cover the three layers this passes
through: the argparse type, the conform.py CLI parser, and conformed_vox_img_size.
"""

import nibabel as nib
import numpy as np
import pytest

from FastSurferCNN.data_loader.conform import conformed_vox_img_size, make_parser
from FastSurferCNN.utils.arg_types import float_gt_zero_and_le_one, vox_size


@pytest.fixture
def noisy_1mm_image() -> nib.MGHImage:
    """A 256^3 LIA image whose header reports a 1mm voxel size only up to float32 noise."""
    affine = np.array([[-1.0000001, 0, 0, 0], [0, 0, 1.0, 0], [0, -0.99999994, 0, 0], [0, 0, 0, 1.0]])
    img = nib.MGHImage(np.zeros((256,) * 3, dtype=np.uint8), affine)
    img.header.set_zooms((np.float32(1.0000001), np.float32(0.99999994), np.float32(1.0)))
    return img


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # 1mm up to float noise, as read from a header -- snapped to exactly 1.0
        ("1.0000001", 1.0), ("0.99999994", 1.0), (1.0000001, 1.0),
        # ... but only around 1mm: hires voxel sizes are passed through unmodified
        ("0.68359375", 0.68359375), ("0.8000005", 0.8000005),
        # the keywords, which are resolved before the numeric conversion
        ("min", "min"), ("any", None), ("none", None), ("infinity", None), (None, None),
    ],
)
def test_vox_size(value, expected):
    """vox_size() tolerates float noise around 1mm without affecting any other input."""
    assert vox_size(value) == expected


@pytest.mark.parametrize("value", ["1.001", "1.1"])
def test_vox_size_rejects_sizes_above_1mm(value):
    """Voxel sizes that are genuinely coarser than 1mm are still rejected."""
    with pytest.raises(ValueError):
        vox_size(value)


def test_float_gt_zero_and_le_one_stays_strict():
    """The shared validator (also used for --robust and --conform_to_1mm_threshold) is unchanged."""
    with pytest.raises(ValueError):
        float_gt_zero_and_le_one("1.0000001")


def test_parser_accepts_noisy_1mm_vox_size():
    """The failing call from recon-surf.sh: conform.py --check_only --vox_size 1.0000001."""
    args = make_parser().parse_args(["-i", "orig.mgz", "--check_only", "--vox_size", "1.0000001", "--dtype", "any"])
    assert args.vox_size == 1.0


@pytest.mark.parametrize("value", ["min", 1.0000001])
def test_conformed_vox_img_size_noisy_1mm(noisy_1mm_image, value):
    """A noisy 1mm image is already conformed, for both --vox_size min and an explicit size."""
    target_vox_size, target_img_size = conformed_vox_img_size(noisy_1mm_image, value, "auto")
    assert target_vox_size == pytest.approx(np.ones(3))
    assert np.array_equal(target_img_size, np.full(3, 256))


def test_conformed_vox_img_size_rejects_sizes_above_1mm(noisy_1mm_image):
    """Voxel sizes coarser than 1mm by more than float noise are still rejected."""
    with pytest.raises(ValueError):
        conformed_vox_img_size(noisy_1mm_image, 1.1, "auto")
