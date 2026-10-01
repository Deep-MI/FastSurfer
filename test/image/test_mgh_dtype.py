"""Tests for the data type of the image files FastSurfer writes.

`MGHHeader.from_header` does not carry the data type over from a non-MGH header: it returns float32
whatever the source says. `as_mgh_image` sets it instead, which is what makes the three rules below
hold.

- The written type does not depend on the container the header came from, nor on the container it is
  written to. The output format only decides which types are available at all.
- A type that cannot be honoured is refused, not quietly replaced: neither one that would round or
  clip the data, nor one the format cannot store. Narrowing is the caller's to do, deliberately and
  outside, because only the caller knows whether to cast, to rescale or to refuse. The single output
  whose type is not ours to choose asks for `storable_dtype` on purpose.
- The aseg files are written as uchar, as FreeSurfer writes them, whatever the type of the
  segmentation they were reduced from.
"""

import nibabel as nib
import numpy as np
import pytest

from FastSurferCNN.data_loader.data_utils import (
    MGH_DTYPES,
    NIFTI_DTYPES,
    as_mgh_image,
    choose_dtype,
    fits_dtype,
    load_maybe_conform,
    save_image,
    storable_dtype,
)
from FastSurferCNN.reduce_to_aseg import create_mask_and_save, reduce_to_aseg_and_save

AFFINE = np.eye(4)
SHAPE = (8, 8, 8)
BOTH_FORMATS = pytest.mark.parametrize("suffix", [".mgz", ".nii.gz"], ids=["mgz", "nii.gz"])


def header_of(dtype, container="mgh"):
    """A header of `dtype`, in either container."""
    data = np.zeros(SHAPE, dtype=dtype)
    if container == "mgh":
        return nib.MGHImage(data, AFFINE).header
    # nibabel refuses to guess int64 from the data, so the type is stated either way
    return nib.Nifti1Image(data, AFFINE, dtype=dtype).header


def written(path):
    """The image at `path`, reloaded, so the assertions are about the file and not the object."""
    return nib.load(path)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32], ids=str)
@pytest.mark.parametrize("source", ["mgh", "nifti"])
@BOTH_FORMATS
def test_dtype_depends_on_neither_container(dtype, source, suffix, tmp_path):
    """The same data and header give the same type, whichever container is on either side."""
    data = np.zeros(SHAPE, dtype=dtype)
    out_file = tmp_path / f"x{suffix}"
    save_image(header_of(dtype, source), AFFINE, data, out_file)

    # MGH stores big-endian, so compare in native byte order
    assert written(out_file).get_data_dtype().newbyteorder("=") == np.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.int64, np.float64], ids=["int64", "float64"])
def test_a_wide_type_survives_where_the_format_allows_it(dtype, tmp_path):
    """NIfTI stores int64 and float64, so neither is narrowed there. The .mgz mirror is below."""
    data = np.full(SHAPE, 2 ** 40 if dtype is np.int64 else 0.1 + 0.2, dtype=dtype)

    out_file = tmp_path / "x.nii.gz"
    save_image(header_of(dtype, "nifti"), AFFINE, data, out_file)

    assert written(out_file).get_data_dtype() == np.dtype(dtype)
    assert np.asarray(written(out_file).dataobj)[0, 0, 0] == data[0, 0, 0]


def probabilities():
    """Soft labels, as the CC module writes them with the header of the conformed image."""
    data = np.zeros(SHAPE, dtype=np.float32)
    data[0, 0, 0] = 0.37
    return data


def label_over_uchar():
    """A label a uchar cannot hold."""
    data = np.zeros(SHAPE, dtype=np.int16)
    data[0, 0, 0] = 500
    return data


@pytest.mark.parametrize(
    ("data", "header_dtype", "explicit", "damage"),
    [
        (probabilities, np.uint8, None, "rounded"),
        (probabilities, np.int16, np.uint8, "rounded"),
        (label_over_uchar, np.uint8, None, "clipped"),
        (label_over_uchar, np.int16, np.uint8, "clipped"),
    ],
    ids=["float via header", "float via dtype", "wide via header", "wide via dtype"],
)
@BOTH_FORMATS
def test_a_type_that_would_lose_data_is_refused(data, header_dtype, explicit, damage, suffix, tmp_path):
    """Both axes, both kinds of loss, both formats.

    The rule is the same whether the type came from the header or from the caller, so the `dtype`
    knob cannot be used to clip either. `data.astype(...)` at the call site is how you ask for that
    on purpose.

    The header's own container is not varied, because `test_dtype_depends_on_neither_container`
    already establishes that it makes no difference.
    """
    with pytest.raises(ValueError, match=f"would be {damage}"):
        save_image(header_of(header_dtype), AFFINE, data(), tmp_path / f"x{suffix}", dtype=explicit)


def test_explicit_dtype_overrides_the_header(tmp_path):
    """What the knob is for: the aseg is uchar even though its header says int16."""
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[0, 0, 0] = 42

    out_file = tmp_path / "explicit.mgz"
    save_image(header_of(np.int16), AFFINE, labels, out_file, dtype=np.uint8)

    assert written(out_file).get_data_dtype() == np.dtype(np.uint8)
    assert np.asarray(written(out_file).dataobj).max() == 42


def int64_of(*values):
    data = np.zeros(SHAPE, dtype=np.int64)
    for i, value in enumerate(values):
        data.flat[i] = value
    return data


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (lambda: int64_of(0, 250), np.int16),
        (lambda: int64_of(-1, 500), np.int16),
        (lambda: int64_of(0, 2 ** 20), np.int32),
        (lambda: np.full(SHAPE, 0.1 + 0.2, np.float64), np.float32),
    ],
    ids=["small", "signed", "large", "float64"],
)
def test_storable_dtype_writes_the_archival_copy(data, expected, tmp_path):
    """mri/rawavg.mgz, whose type belongs to whoever produced the input file.

    int64 is the default integer width in numpy and float64 is what a scaled NIfTI reads back as,
    and MGH stores neither. Such data has to land on a type the format has: the narrowest that holds
    every value, and of the same kind, so a float stays a float and signed stays signed. Values in 0
    to 250 would fit uchar, but changing the signedness of someone else's data is a bigger liberty
    than a wider file.
    """
    array = data()
    out_file = tmp_path / "001.mgz"

    save_image(header_of(array.dtype, "nifti"), AFFINE, array, out_file,
               dtype=storable_dtype(array))

    assert written(out_file).get_data_dtype().newbyteorder("=") == np.dtype(expected)
    if np.issubdtype(array.dtype, np.integer):
        assert set(np.asarray(written(out_file).dataobj).flatten().tolist()) == set(array.flatten().tolist())


@pytest.mark.parametrize("wanted", [np.int64, np.float64], ids=["int64", "float64"])
def test_a_type_mgh_cannot_store_is_refused(wanted, tmp_path):
    """MGH has no int64 and no float64, so it says so rather than picking something else.

    The format is asked rather than checked against a list, so this also covers a type nobody
    thought to enumerate. A float64 request answered with an integer would not be a narrower version
    of the request but a different file; callers that do not choose their own type ask for
    `storable_dtype` instead.
    """
    data = np.zeros(SHAPE, dtype=wanted)

    with pytest.raises(ValueError, match="cannot store"):
        save_image(header_of(wanted, "nifti"), AFFINE, data, tmp_path / "x.mgz", dtype=wanted)


def test_storable_dtype_keeps_the_kind_and_the_values():
    """The archival copy of the input, whose type was chosen by whoever produced the file."""
    for dtype in (np.uint8, np.int16, np.int32, np.float32):
        own = np.zeros(SHAPE, dtype)
        assert storable_dtype(own) == np.dtype(dtype), "a storable type is kept as it is"

    # MGH has no float64, and the answer must still be a float rather than the narrowest that fits
    assert storable_dtype(np.zeros(SHAPE, np.float64)) == np.dtype(np.float32)
    # nor is an integer request answered with a float
    assert np.issubdtype(storable_dtype(np.full(SHAPE, 300, np.int64)), np.integer)
    assert fits_dtype(np.full(SHAPE, 300, np.int64), storable_dtype(np.full(SHAPE, 300, np.int64)))

    with pytest.raises(ValueError, match="cannot store"):
        storable_dtype(np.full(SHAPE, 2 ** 40, np.int64))


@pytest.mark.parametrize("conf_name", ["orig.mgz", "orig.nii.gz"], ids=["mgz", "nii.gz"])
def test_the_image_handed_back_is_the_one_written(conf_name, tmp_path):
    """save_image returns what it wrote, so load_maybe_conform's caller gets the right container."""
    affine = np.diag([0.8, 0.8, 0.8, 1.0])
    affine[:3, 3] = -128.0
    voxels = np.zeros((64, 64, 64), dtype=np.uint8)
    nib.save(nib.Nifti1Image(voxels, affine), tmp_path / "T1.nii.gz")

    out_file, img, _ = load_maybe_conform(
        tmp_path / conf_name, tmp_path / "T1.nii.gz",
        vox_size=1.0, img_size="auto", orientation="lia", order=1, dtype=np.uint8,
    )

    assert out_file.name.endswith(conf_name.split(".", 1)[1])
    assert isinstance(img, nib.MGHImage) == conf_name.endswith(".mgz")


def test_an_integer_nifti_carries_no_scale_factor(tmp_path):
    """Left free, nibabel may add one, and a label read back through a scale is no longer an integer.

    Pinned rather than assumed, because the slope is only visible in the file: nibabel folds it into
    the array proxy on load and blanks the header field.
    """
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[0, 0, 0] = 253

    out_file = tmp_path / "labels.nii.gz"
    save_image(header_of(np.int16, "nifti"), AFFINE, labels, out_file)

    assert written(out_file).dataobj.slope == 1.0
    assert written(out_file).dataobj.inter == 0.0
    assert np.asarray(written(out_file).dataobj)[0, 0, 0] == 253


def test_header_is_required():
    """The affine covers the geometry, but only the header carries the acquisition parameters.

    Every caller has one, so it is required rather than optional and a caller with no source image
    has to say so by passing an `MGHHeader()`.
    """
    with pytest.raises(TypeError):
        as_mgh_image(np.zeros(SHAPE, dtype=np.uint8), AFFINE)


def test_choose_dtype():
    """The decision behind every writer, without going through a file."""
    labels = np.zeros(SHAPE, np.int16)
    probabilities = np.zeros(SHAPE, np.float32)

    assert choose_dtype(labels, header_of(np.int16)) == np.dtype(np.int16), "the header holds it"
    assert choose_dtype(labels, header_of(np.int16), np.uint8) == np.dtype(np.uint8), "narrowed on request"
    # the byte order belongs to the format, not to the decision
    assert choose_dtype(labels, header_of(np.int16)).byteorder in "=|"

    with pytest.raises(ValueError, match="would be rounded"):
        choose_dtype(probabilities, header_of(np.uint8))
    with pytest.raises(ValueError, match="would be rounded"):
        choose_dtype(probabilities, header_of(np.int16), np.uint8)
    with pytest.raises(ValueError, match="would be clipped"):
        choose_dtype(np.full(SHAPE, 500, np.int16), header_of(np.uint8))


def test_fits_dtype():
    """The shared rule the writers decide by."""
    assert fits_dtype(np.zeros(SHAPE, np.int16), np.uint8), "in range"
    assert not fits_dtype(np.full(SHAPE, 500, np.int16), np.uint8), "above the range"
    assert not fits_dtype(np.full(SHAPE, -1, np.int16), np.uint8), "below the range"
    assert not fits_dtype(np.zeros(SHAPE, np.float32), np.uint8), "a float would be rounded"
    assert fits_dtype(np.zeros(SHAPE, np.uint8), np.int16), "widening always fits"
    assert fits_dtype(np.zeros(SHAPE, np.int16), np.float32), "a float holds a small integer"
    assert fits_dtype(np.zeros(SHAPE, np.float64), np.float32), "a float may lose only precision"
    assert fits_dtype(np.zeros((0,), np.int16), np.uint8), "an empty array has nothing to clip"
    assert fits_dtype(np.zeros(SHAPE, bool), np.uint8), "a bool is 0 or 1, so it always fits"
    # an integer rounded into a float loses its identity, so a float target has a range too
    exact = np.full(SHAPE, 2 ** 24, dtype=np.int64)
    assert fits_dtype(exact, np.float32), "float32 counts every integer up to 2**24"
    assert not fits_dtype(exact + 1, np.float32), "and not the one after it"
    assert fits_dtype(exact + 1, np.float64), "float64 counts far past it"


def test_only_the_type_changes_not_the_rest_of_the_header(tmp_path):
    """Setting the type must not cost the header. Only the type is ours to override."""
    source = nib.MGHImage(np.zeros(SHAPE, dtype=np.uint8), AFFINE)
    # the keys are MGH header field names, hence the ignore for the echo time one
    acquisition = {"tr": 2300.0, "te": 2.98, "ti": 900.0, "flip_angle": 0.15708}  # codespell:ignore te
    for field, value in acquisition.items():
        source.header[field] = value

    soft_labels = np.zeros(SHAPE, dtype=np.float32)
    soft_labels[0, 0, 0] = 0.37
    out_file = tmp_path / "soft.mgz"
    nib.save(as_mgh_image(soft_labels, AFFINE, source.header, dtype=np.float32), out_file)

    assert written(out_file).get_data_dtype() == np.dtype(">f4"), "the requested type is used"
    for field, value in acquisition.items():
        assert float(written(out_file).header[field]) == pytest.approx(value), f"{field} survives"
    assert np.allclose(written(out_file).affine, AFFINE)


def test_the_caller_header_is_not_modified():
    """The aseg and the mask are written from one header on two threads, so it must not be touched."""
    header = header_of(np.uint8)
    before = (header.get_data_dtype(), float(header["fov"]))

    as_mgh_image(np.zeros(SHAPE, dtype=np.uint8), AFFINE, header, dtype=np.int16)

    assert (header.get_data_dtype(), float(header["fov"])) == before


def dkt_segmentation():
    """A DKT segmentation as FastSurfer writes it: int16, with cortical labels in the 1000s."""
    seg = np.zeros((16, 16, 16), dtype=np.int16)
    seg[0], seg[1], seg[2], seg[3] = 1035, 2035, 251, 17
    return seg


def test_aseg_is_written_as_uchar(tmp_path):
    """FreeSurfer writes aseg.auto.mgz as uchar, so FastSurfer does too."""
    seg = dkt_segmentation()
    header = nib.MGHImage(seg, AFFINE).header
    assert header.get_data_dtype() == np.dtype(">i2"), "the input really is int16"

    out_file = tmp_path / "aseg.auto.mgz"
    reduce_to_aseg_and_save(seg, AFFINE, header, out_file)

    assert written(out_file).get_data_dtype() == np.dtype(np.uint8)
    # the labels survive the narrowing: cortex became 3 and 42, the rest is unchanged
    assert set(np.unique(np.asarray(written(out_file).dataobj))) == {0, 3, 17, 42, 251}


def test_mask_is_written_as_uchar(tmp_path):
    """The mask carries the same uchar guarantee as the aseg, not the type it was derived from."""
    seg = np.zeros((16, 16, 16), dtype=np.int16)
    seg[4:12, 4:12, 4:12] = 17
    header = nib.MGHImage(seg, AFFINE).header

    out_file = tmp_path / "mask.mgz"
    create_mask_and_save(seg, AFFINE, header, out_file)

    assert written(out_file).get_data_dtype() == np.dtype(np.uint8)


def label_above_255():
    seg = dkt_segmentation()
    seg[4] = 500
    return seg


def negative_label():
    seg = dkt_segmentation()
    seg[4] = -1
    return seg


@pytest.mark.parametrize("case", [label_above_255, negative_label], ids=["above 255", "negative"])
def test_an_aseg_uchar_would_damage_is_refused(case, tmp_path):
    """reduce_to_aseg asks for uchar because an aseg has no label outside it.

    A label that does not fit means something went wrong upstream, so it surfaces rather than being
    papered over by writing a type FreeSurfer does not expect here.
    """
    header = nib.MGHImage(case(), AFFINE).header

    with pytest.raises(ValueError, match="would be clipped"):
        reduce_to_aseg_and_save(case(), AFFINE, header, tmp_path / "wide.mgz")


def test_conformed_copy_of_a_float_input_is_not_float(tmp_path):
    """The orig.10mm.mgz path: conform is asked for uchar, so a float input still lands as uchar."""
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

    assert written(out_file).get_data_dtype() == np.dtype(np.uint8)


@pytest.mark.parametrize(
    ("candidates", "container"), [(MGH_DTYPES, nib.MGHImage), (NIFTI_DTYPES, nib.Nifti1Image)],
    ids=["mgz", "nii.gz"],
)
def test_the_dtype_lists_are_what_the_format_really_takes(candidates, container):
    """The lists decide what is refused, so a missing entry refuses something that would have worked.

    Checked against nibabel rather than trusted, because that is the only authority on it.
    """
    every = (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
             np.uint64, np.int64, np.float32, np.float64)
    accepted = set()
    for dtype in every:
        img = container(np.zeros((2, 2, 2), np.uint8), AFFINE)
        try:
            img.set_data_dtype(dtype)
            accepted.add(np.dtype(dtype))
        except Exception:  # noqa: BLE001  the format refusing is the answer we want
            pass

    assert {np.dtype(c) for c in candidates} == accepted
