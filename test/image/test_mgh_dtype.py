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

from FastSurferCNN.data_loader.data_utils import (
    as_mgh_image,
    fits_dtype,
    load_maybe_conform,
    save_image,
)
from FastSurferCNN.reduce_to_aseg import create_mask_and_save, reduce_to_aseg_and_save

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


@pytest.mark.parametrize("with_header", [True, False], ids=["from a header", "headerless"])
def test_unstorable_dtype_falls_back_instead_of_raising(with_header, caplog):
    """MGH has no float64. That must degrade to float32 with a warning, not abort a run.

    The headerless case is the one that bites: nibabel raises while constructing the image, before
    any fallback of ours could run.
    """
    data = np.zeros(SHAPE, dtype=np.float64)
    data[0, 0, 0] = 0.37
    header = nib.Nifti1Image(data, AFFINE).header if with_header else None

    img = as_mgh_image(data, AFFINE, header)

    assert img.get_data_dtype() == np.dtype(">f4")
    assert "cannot store" in caplog.text


@pytest.mark.parametrize("header_dtype", [np.uint8, np.int16], ids=["uint8", "int16"])
@pytest.mark.parametrize("container", ["nifti", "mgh", "none"])
def test_float_data_is_never_stored_as_an_integer(container, header_dtype, tmp_path):
    """Probabilities must survive, whatever integer type the header carries.

    The CC module writes its soft labels with the header of the conformed image, which is uchar, so
    0.37 was stored as 0 and the probability map came back as a binary mask. An MGH header is the
    case that matters: nibabel applies its type while constructing the image, not only afterwards.
    """
    soft_labels = np.zeros(SHAPE, dtype=np.float32)
    soft_labels[0, 0, 0] = 0.37
    integers = np.zeros(SHAPE, dtype=header_dtype)
    headers = {
        "nifti": nib.Nifti1Image(integers, AFFINE).header,
        "mgh": nib.MGHImage(integers, AFFINE).header,
        "none": None,
    }

    out_file = tmp_path / "soft.mgz"
    nib.save(as_mgh_image(soft_labels, AFFINE, headers[container]), out_file)

    written = nib.load(out_file)
    assert written.get_data_dtype() == np.dtype(">f4")
    assert np.asarray(written.dataobj)[0, 0, 0] == pytest.approx(0.37)


@pytest.mark.parametrize("container", ["nifti", "mgh"])
def test_integer_data_wider_than_the_header_is_not_clipped(container, tmp_path):
    """The mirror of the float case: a header narrower than its data must not silently clip it.

    A uchar header with int16 data holding 500 wrote 255, losing the label. The NIfTI path used to
    escape it only because the type was dropped entirely and everything became float32.
    """
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[0, 0, 0] = 500
    narrow = np.zeros(SHAPE, dtype=np.uint8)
    header = {
        "nifti": nib.Nifti1Image(narrow, AFFINE).header,
        "mgh": nib.MGHImage(narrow, AFFINE).header,
    }[container]

    out_file = tmp_path / "labels.mgz"
    nib.save(as_mgh_image(labels, AFFINE, header), out_file)

    written = nib.load(out_file)
    assert written.get_data_dtype() != np.dtype(np.uint8)
    assert np.asarray(written.dataobj).max() == 500


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 500], np.dtype(">u2")),
        ([-1, 500], np.dtype(">i2")),
        ([0, 2 ** 20], np.dtype(">i4")),
    ],
    ids=["fits uint16", "needs a sign", "needs int32"],
)
def test_widening_picks_a_type_mgh_can_store(values, expected, tmp_path):
    """int64 is the default integer width here, and MGH cannot store it.

    Widening to the array's own type therefore hit the MGHError fallback, which handed the type back
    to the narrow header and clipped after all: 500 was written as 255, with two log lines saying
    first that clipping had been avoided and then that it had not.
    """
    labels = np.zeros(SHAPE, dtype=np.int64)
    labels[0, 0, 0], labels[0, 0, 1] = values
    header = nib.MGHImage(np.zeros(SHAPE, dtype=np.uint8), AFFINE).header

    out_file = tmp_path / "labels.mgz"
    nib.save(as_mgh_image(labels, AFFINE, header), out_file)

    written = nib.load(out_file)
    assert written.get_data_dtype() == expected
    assert set(np.asarray(written.dataobj).flatten().tolist()) == {0, *values}


def test_prefer_dtype_is_ignored_when_it_would_lose_data(tmp_path):
    """The narrowing knob is a preference, not a force, so it cannot be used to clip."""
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[0, 0, 0] = 500
    header = nib.MGHImage(labels, AFFINE).header

    fitting = as_mgh_image(np.zeros(SHAPE, dtype=np.int16), AFFINE, header, prefer_dtype=np.uint8)
    assert fitting.get_data_dtype() == np.dtype(np.uint8), "honoured where the data fits"

    out_file = tmp_path / "labels.mgz"
    nib.save(as_mgh_image(labels, AFFINE, header, prefer_dtype=np.uint8), out_file)
    written = nib.load(out_file)
    assert written.get_data_dtype() != np.dtype(np.uint8)
    assert np.asarray(written.dataobj).max() == 500


def test_explicit_dtype_says_when_it_clips(tmp_path, caplog):
    """save_image still forces the type, since run_prediction narrows with it, but no longer silently."""
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[0, 0, 0] = 500
    header = nib.MGHImage(labels, AFFINE).header

    out_file = tmp_path / "forced.mgz"
    save_image(header, AFFINE, labels, out_file, dtype=np.uint8)

    assert nib.load(out_file).get_data_dtype() == np.dtype(np.uint8), "the caller's decision stands"
    assert "does not fit" in caplog.text


def test_fits_dtype():
    """The shared rule the writers narrow by."""
    assert fits_dtype(np.zeros(SHAPE, np.int16), np.uint8), "in range"
    assert not fits_dtype(np.full(SHAPE, 500, np.int16), np.uint8), "above the range"
    assert not fits_dtype(np.full(SHAPE, -1, np.int16), np.uint8), "below the range"
    assert not fits_dtype(np.zeros(SHAPE, np.float32), np.uint8), "a float would be rounded"
    assert fits_dtype(np.zeros(SHAPE, np.uint8), np.int16), "widening always fits"
    assert fits_dtype(np.zeros(SHAPE, np.int16), np.float32), "a float target holds any integer"
    assert fits_dtype(np.zeros((0,), np.int16), np.uint8), "an empty array has nothing to clip"


def test_only_the_type_changes_not_the_rest_of_the_header(tmp_path):
    """Widening the type must not cost the header. Only the type is ours to override."""
    source = nib.MGHImage(np.zeros(SHAPE, dtype=np.uint8), AFFINE)
    # the keys are MGH header field names, hence the ignore for the echo time one
    acquisition = {"tr": 2300.0, "te": 2.98, "ti": 900.0, "flip_angle": 0.15708}  # codespell:ignore te
    for field, value in acquisition.items():
        source.header[field] = value

    soft_labels = np.zeros(SHAPE, dtype=np.float32)
    soft_labels[0, 0, 0] = 0.37
    out_file = tmp_path / "soft.mgz"
    nib.save(as_mgh_image(soft_labels, AFFINE, source.header), out_file)

    written = nib.load(out_file)
    assert written.get_data_dtype() == np.dtype(">f4"), "the type is widened"
    for field, value in acquisition.items():
        assert float(written.header[field]) == pytest.approx(value), f"{field} survives"
    assert np.allclose(written.affine, AFFINE)


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


def test_mask_is_written_as_uchar(tmp_path):
    """The mask carries the same uchar guarantee as the aseg, not the type it was derived from."""
    seg = np.zeros((16, 16, 16), dtype=np.int16)
    seg[4:12, 4:12, 4:12] = 17
    header = nib.MGHImage(seg, AFFINE).header

    out_file = tmp_path / "mask.mgz"
    create_mask_and_save(seg, AFFINE, header, out_file)

    assert nib.load(out_file).get_data_dtype() == np.dtype(np.uint8)


def label_above_255():
    seg = dkt_segmentation()
    seg[4] = 500
    return seg, 500.0


def negative_label():
    seg = dkt_segmentation()
    seg[4] = -1
    return seg, -1.0


def fractional_float():
    # 17.5 is exact in float32, so the assertion tests the narrowing and not float promotion
    seg = dkt_segmentation().astype(np.float32)
    seg[4] = 17.5
    return seg, 17.5


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
