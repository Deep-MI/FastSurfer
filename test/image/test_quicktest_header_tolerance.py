"""Tests for the header comparison the quicktest suite uses, `quicktest.helper`.

They live here rather than next to the code they test because `test/quicktest/conftest.py` requires
`REF_DIR` and `SUBJECTS_DIR` at import, so nothing in that directory can be collected without a
reference dataset, while `test/image` runs in the unittest workflow on every push.

What is pinned: below 1mm the direction cosines are computed rather than snapped, so `Mdc` carries
float dust of the order of 1e-18 that no rerun reproduces bit for bit. Comparing headers exactly
therefore failed every non-1mm file forever. Only float fields are relaxed, so a changed data type
or image size is still reported.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quicktest.helper import assert_same_headers, equal_within_tolerance  # noqa: E402

# the largest deviation measured across all 39 shared 0.8mm files of a released v2.5.4 run
MDC_DUST = 8.842e-18


def header_of(shape=(320, 320, 320), vox_size=0.8, dtype=np.uint8):
    """The header of an empty LIA-ish image, as the suite reads it back from an .mgz."""
    affine = np.diag([vox_size, vox_size, vox_size, 1.0])
    affine[:3, 3] = -0.5 * shape[0] * vox_size
    return nib.MGHImage(np.zeros(shape, dtype=dtype), affine).header


def test_mdc_float_dust_passes():
    """The failure this fixes: an Mdc that differs only in the last bits of a computed cosine."""
    reference, test = header_of(), header_of()
    mdc = np.array(reference["Mdc"])
    mdc[0, 1] += MDC_DUST
    test["Mdc"] = mdc

    assert not np.array_equal(reference["Mdc"], test["Mdc"]), "the two headers must really differ"
    assert_same_headers(test, reference)


def test_real_orientation_difference_still_fails():
    """A tolerance that also hid a genuinely different orientation would be worthless."""
    reference, test = header_of(), header_of()
    mdc = np.array(reference["Mdc"])
    mdc[0, 1] += 0.5
    test["Mdc"] = mdc

    with pytest.raises(BaseException, match="Mdc"):
        assert_same_headers(test, reference)


def test_fov_difference_still_fails():
    """fov=0 against a real extent is the bug found in the released images, not float noise."""
    reference, test = header_of(), header_of()
    reference["fov"] = 256.0
    test["fov"] = 0.0

    with pytest.raises(BaseException, match="fov"):
        assert_same_headers(test, reference)


def test_data_type_difference_still_fails():
    """Non-float fields keep the exact comparison; 3 of 37 1.0mm files differ in type alone."""
    with pytest.raises(BaseException, match="type"):
        assert_same_headers(header_of(dtype=np.int16), header_of(dtype=np.uint8))


@pytest.mark.parametrize(
    ("reference", "test", "expected"),
    [
        # float fields are compared with the tolerance
        (np.float32([1.0, 0.0]), np.float32([1.0, MDC_DUST]), True),
        (np.float32([1.0, 0.0]), np.float32([1.0, 0.5]), False),
        # integers are not, so a size or count is never accepted as noise
        (np.int32([256, 256]), np.int32([256, 257]), False),
        # nor is anything of another shape, or not a number at all
        (np.float32([1.0, 0.0]), np.float32([1.0, 0.0, 0.0]), False),
        ("MGH", "MGH", False),
    ],
)
def test_equal_within_tolerance(reference, test, expected):
    """Only float fields of matching shape are eligible for the tolerance."""
    assert equal_within_tolerance(reference, test, rtol=1e-6, atol=1e-6) is expected
