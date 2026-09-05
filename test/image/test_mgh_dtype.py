"""Regression tests for the data type of the .mgz files FastSurfer writes.

`MGHHeader.from_header` does not carry the data type over from a non-MGH header: it returns float32
whatever the source says. Every .mgz written with a header that came from a .nii was therefore stored
as float32, which is how the 1mm copy CerebNet conforms for a high-res subject became a float32 file
four times the size of the uint8 one it asked for, holding nothing but integers.

The rule pinned here is that the written type must not depend on the container the header came from.

The aseg files are the second half of the same problem: they inherit the header of the int16
segmentation they are reduced from, where FreeSurfer, and FastSurfer up to v2.3.3, write uchar.
"""

import nibabel as nib
import numpy as np
import pytest

from FastSurferCNN.data_loader.data_utils import as_mgh_image, load_maybe_conform, save_image
from FastSurferCNN.reduce_to_aseg import reduce_to_aseg_and_save

AFFINE = np.eye(4)
SHAPE = (8, 8, 8)


def headers_of(dtype):
    """The same image as a NIfTI and as an MGH header, plus no header at all."""
    data = np.zeros(SHAPE, dtype=dtype)
    return {
        "nifti": nib.Nifti1Image(data, AFFINE).header,
        "mgh": nib.MGHImage(data, AFFINE).header,
        "none": None,
    }


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32], ids=str)
@pytest.mark.parametrize("source", ["nifti", "mgh", "none"])
def test_dtype_does_not_depend_on_the_source_container(dtype, source):
    """The bug: a NIfTI header silently produced float32, an MGH header did not."""
    data = np.zeros(SHAPE, dtype=dtype)
    img = as_mgh_image(data, AFFINE, headers_of(dtype)[source])
    # MGH stores big-endian, so compare in native byte order
    assert img.get_data_dtype().newbyteorder("=") == np.dtype(dtype)


def test_unstorable_dtype_falls_back_instead_of_raising(caplog):
    """MGH has no float64. That must degrade to float32 with a warning, not abort a run."""
    data = np.zeros(SHAPE, dtype=np.float64)
    header = nib.Nifti1Image(data, AFFINE).header

    img = as_mgh_image(data, AFFINE, header)

    assert img.get_data_dtype() == np.dtype(">f4")
    assert "cannot store" in caplog.text


def test_explicit_dtype_still_wins(tmp_path):
    """save_image(dtype=...) overrides whatever the header carried."""
    data = np.zeros(SHAPE, dtype=np.float32)
    header = nib.Nifti1Image(data, AFFINE).header

    out_file = tmp_path / "explicit.mgz"
    save_image(header, AFFINE, data, out_file, dtype=np.uint8)

    assert nib.load(out_file).get_data_dtype() == np.dtype(np.uint8)


def dkt_segmentation():
    """A DKT segmentation as FastSurfer writes it: int16, with cortical labels in the 1000s."""
    seg = np.zeros((16, 16, 16), dtype=np.int16)
    seg[0], seg[1], seg[2], seg[3] = 1035, 2035, 251, 17
    return seg


def test_aseg_is_written_as_uchar(tmp_path):
    """FreeSurfer writes aseg.auto.mgz as uchar, and so did FastSurfer before the int16 carried over."""
    seg = dkt_segmentation()
    header = nib.MGHImage(seg, AFFINE).header
    assert header.get_data_dtype() == np.dtype(">i2"), "the input really is int16"

    out_file = tmp_path / "aseg.auto.mgz"
    reduce_to_aseg_and_save(seg, AFFINE, header, out_file)

    written = nib.load(out_file)
    assert written.get_data_dtype() == np.dtype(np.uint8)
    # the labels survive the narrowing: cortex became 3 and 42, the rest is unchanged
    assert set(np.unique(np.asarray(written.dataobj))) == {0, 3, 17, 42, 251}


def label_above_255():
    seg = dkt_segmentation()
    seg[4] = 500
    return seg, 500.0


def negative_label():
    seg = dkt_segmentation()
    seg[4] = -1
    return seg, -1.0


def fractional_float():
    seg = dkt_segmentation().astype(np.float32)
    seg[4] = 17.6
    return seg, 17.6


@pytest.mark.parametrize(
    "case", [label_above_255, negative_label, fractional_float],
    ids=["above 255", "negative", "fractional float"],
)
def test_data_that_uchar_would_damage_is_not_narrowed(case, tmp_path):
    """Narrowing must never round or clip. Only integer labels within 0 to 255 are eligible.

    Without the range and dtype check, uchar turned 17.6 into 18 and -1 into 0, silently.
    """
    seg, sentinel = case()
    header = nib.MGHImage(seg, AFFINE).header

    out_file = tmp_path / "wide.mgz"
    reduce_to_aseg_and_save(seg, AFFINE, header, out_file)

    written = nib.load(out_file)
    assert written.get_data_dtype() != np.dtype(np.uint8)
    assert sentinel in np.asarray(written.dataobj)


def test_conformed_copy_of_a_float_input_is_not_float(tmp_path):
    """The path that produced orig.10mm.mgz: a float NIfTI input, uint8 asked for, uint8 expected."""
    affine = np.diag([0.8, 0.8, 0.8, 1.0])
    affine[:3, 3] = -128.0
    voxels = np.rint(np.random.default_rng(0).random((64, 64, 64)) * 255).astype(np.float32)
    nib.save(nib.Nifti1Image(voxels, affine), tmp_path / "T1.nii.gz")

    out_file, _, _ = load_maybe_conform(
        tmp_path / "orig.mgz",
        tmp_path / "T1.nii.gz",
        vox_size=1.0,
        img_size="auto",
        orientation="lia",
        order=1,
        dtype=np.uint8,
    )

    assert nib.load(out_file).get_data_dtype() == np.dtype(np.uint8)
