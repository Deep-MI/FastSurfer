"""Tests for the input copies FastSurfer places in the subject directory.

``mri/orig/001.<ext>`` is a byte-for-byte copy of the input, so a re-encoding of it would not do:
a scaled NIfTI carries its scale factor and its own data type, and an .mgz cannot hold either.

``mri/rawavg.mgz`` is the copy FreeSurfer's tools read, ``pctsurfcon`` in particular, which builds
that path itself. It has to be an MGH file whatever arrived, and it has to carry the field of view,
which a plain nibabel write leaves at 0.
"""

import filecmp

import nibabel as nib
import numpy as np
import pytest

from FastSurferCNN.copy_input import (
    RAWAVG_PATH,
    T2_RAWAVG_PATH,
    archive_input,
    image_suffix,
    main,
    write_rawavg,
)

SHAPE = (12, 14, 16)


def scaled_nifti(path, dtype=np.int16, slope=0.001):
    """A NIfTI that reads back as float64, as scanner conversion routinely produces."""
    affine = np.diag([0.8, 0.9, 1.0, 1.0])
    data = np.arange(np.prod(SHAPE), dtype=dtype).reshape(SHAPE)
    image = nib.Nifti1Image(data, affine)
    image.header.set_slope_inter(slope, 0.0)
    nib.save(image, path)
    return path


def mgh(path, dtype=np.uint8):
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    data = np.arange(np.prod(SHAPE), dtype=np.int64).reshape(SHAPE) % 200
    nib.save(nib.MGHImage(data.astype(dtype), affine), path)
    return path


@pytest.mark.parametrize("maker", [scaled_nifti, mgh], ids=["nii.gz", "mgz"])
def test_the_archival_copy_is_byte_for_byte(maker, tmp_path):
    """Nothing about the input may change, including the parts an .mgz could not represent."""
    source = maker(tmp_path / f"input{'.nii.gz' if maker is scaled_nifti else '.mgz'}")

    copy = archive_input(source, tmp_path / "sub" / "mri" / "orig")

    assert copy.name == f"001{image_suffix(source)}"
    assert filecmp.cmp(source, copy, shallow=False), "the copy differs from the input"


def test_rawavg_from_a_nifti_is_a_converted_mgz(tmp_path):
    """The values carry over and the field of view is set, which a plain nibabel write leaves at 0."""
    source = scaled_nifti(tmp_path / "input.nii.gz")
    copy = archive_input(source, tmp_path / "sub" / "mri" / "orig")
    rawavg = tmp_path / "sub" / "mri" / RAWAVG_PATH

    write_rawavg(source, rawavg, archive=copy)

    written = nib.load(rawavg)
    assert not rawavg.is_symlink(), "a converted rawavg is a real file"
    assert written.get_data_dtype() == np.dtype(">f4"), "float64 is not storable, float32 is"
    expected_fov = max(n * z for n, z in zip(SHAPE, written.header.get_zooms()[:3], strict=True))
    assert float(written.header["fov"]) == pytest.approx(expected_fov)
    assert np.allclose(np.asanyarray(written.dataobj), np.asanyarray(nib.load(source).dataobj))


def test_rawavg_from_an_mgz_is_a_relative_symlink(tmp_path):
    """The bytes are already right, so a second copy would only be a second copy."""
    source = mgh(tmp_path / "input.mgz")
    copy = archive_input(source, tmp_path / "sub" / "mri" / "orig")
    rawavg = tmp_path / "sub" / "mri" / RAWAVG_PATH

    write_rawavg(source, rawavg, archive=copy)

    assert rawavg.is_symlink()
    assert rawavg.readlink().as_posix() == "orig/001.mgz", "relative, so the directory can move"
    assert filecmp.cmp(source, rawavg, shallow=False)


def test_rerunning_replaces_an_existing_rawavg(tmp_path):
    """A second run must not fail on the symlink or the file the first one left."""
    mgz = mgh(tmp_path / "a.mgz")
    nifti = scaled_nifti(tmp_path / "b.nii.gz")
    rawavg = tmp_path / "sub" / "mri" / RAWAVG_PATH

    write_rawavg(mgz, rawavg, archive=archive_input(mgz, tmp_path / "sub" / "mri" / "orig"))
    assert rawavg.is_symlink()
    # a second subject whose input is a NIfTI, so the symlink has to give way to a real file
    write_rawavg(nifti, rawavg, archive=archive_input(nifti, tmp_path / "other" / "orig"))
    assert not rawavg.is_symlink()
    # and back again
    write_rawavg(mgz, rawavg, archive=archive_input(mgz, tmp_path / "third" / "orig"))
    assert rawavg.is_symlink()


def test_a_conflicting_archive_stops_the_run(tmp_path):
    """Two inputs cannot share one subject directory, and nothing could tell their copies apart."""
    assert main(t1=mgh(tmp_path / "a.mgz"), sd=tmp_path, sid="sub") == 0
    assert main(t1=scaled_nifti(tmp_path / "b.nii.gz"), sd=tmp_path, sid="sub") == 1

    orig_dir = tmp_path / "sub" / "mri" / "orig"
    assert [p.name for p in orig_dir.iterdir()] == ["001.mgz"], "the first archive is untouched"
    # the same input again is a re-run, not a conflict
    assert main(t1=tmp_path / "a.mgz", sd=tmp_path, sid="sub") == 0


def test_a_different_image_under_the_same_extension_stops_the_run(tmp_path):
    """The extension says nothing about the image, so the archive is compared byte for byte."""
    assert main(t1=mgh(tmp_path / "a.mgz"), sd=tmp_path, sid="sub") == 0

    assert main(t1=mgh(tmp_path / "b.mgz", dtype=np.int16), sd=tmp_path, sid="sub") == 1
    assert filecmp.cmp(tmp_path / "a.mgz", tmp_path / "sub" / "mri" / "orig" / "001.mgz",
                       shallow=False), "the first archive is untouched"


def test_rawavg_only_writes_no_archive(tmp_path):
    """What --base and --long need: their input is built by the pipeline, not passed by the user."""
    t1 = scaled_nifti(tmp_path / "t1.nii.gz")

    assert main(t1=t1, sd=tmp_path, sid="sub", rawavg_only=True) == 0

    mri = tmp_path / "sub" / "mri"
    assert (mri / RAWAVG_PATH).is_file()
    assert not (mri / "orig").exists()


def test_rawavg_only_still_archives_a_t2(tmp_path):
    """A longitudinal time point's T1 is built by the pipeline, but its T2 is the user's input."""
    t1 = scaled_nifti(tmp_path / "t1.nii.gz")
    t2 = scaled_nifti(tmp_path / "t2.nii.gz")

    assert main(t1=t1, sd=tmp_path, sid="sub", t2=t2, rawavg_only=True) == 0

    orig_dir = tmp_path / "sub" / "mri" / "orig"
    assert sorted(p.name for p in orig_dir.iterdir()) == ["T2raw.mgz", "T2raw.nii.gz"]
    assert filecmp.cmp(t2, orig_dir / "T2raw.nii.gz", shallow=False)


def test_every_part_of_a_multi_file_image_is_copied(tmp_path):
    """An Analyze .hdr without its .img is not an image, so both are copied and both keep the stem."""
    data = np.arange(np.prod(SHAPE), dtype=np.int16).reshape(SHAPE)
    nib.save(nib.Nifti1Pair(data, np.diag([1.0, 1.0, 1.0, 1.0])), tmp_path / "input.img")

    assert main(t1=tmp_path / "input.img", sd=tmp_path, sid="sub") == 0

    orig_dir = tmp_path / "sub" / "mri" / "orig"
    assert sorted(p.name for p in orig_dir.iterdir()) == ["001.hdr", "001.img"]
    assert np.array_equal(np.asanyarray(nib.load(orig_dir / "001.img").dataobj), data)
    # and a second run of the same input recognises both parts rather than reporting a conflict
    assert main(t1=tmp_path / "input.hdr", sd=tmp_path, sid="sub") == 0


def test_a_nifti_t2_is_archived_and_converted(tmp_path):
    """N4 reads mri/orig/T2raw.mgz, the path recon-all also converts a -T2 input to."""
    t1 = mgh(tmp_path / "t1.mgz")
    t2 = scaled_nifti(tmp_path / "t2.nii.gz")

    assert main(t1=t1, sd=tmp_path, sid="sub", t2=t2) == 0

    mri = tmp_path / "sub" / "mri"
    assert filecmp.cmp(t1, mri / "orig" / "001.mgz", shallow=False)
    assert filecmp.cmp(t2, mri / "orig" / "T2raw.nii.gz", shallow=False)
    assert (mri / RAWAVG_PATH).is_symlink(), "the T1 was an mgz"
    assert nib.load(mri / T2_RAWAVG_PATH).get_data_dtype() == np.dtype(">f4")


def test_rerunning_a_nifti_t2_is_not_a_conflict(tmp_path):
    """Its rawavg is written into mri/orig too, so it matches the same names as the archive."""
    t1 = mgh(tmp_path / "t1.mgz")
    t2 = scaled_nifti(tmp_path / "t2.nii.gz")

    assert main(t1=t1, sd=tmp_path, sid="sub", t2=t2) == 0
    orig_dir = tmp_path / "sub" / "mri" / "orig"
    assert sorted(p.name for p in orig_dir.iterdir()) == ["001.mgz", "T2raw.mgz", "T2raw.nii.gz"]

    assert main(t1=t1, sd=tmp_path, sid="sub", t2=t2) == 0, "the same inputs again is a re-run"
    # but a different T2 still is a conflict, and the converted rawavg must not hide the old archive
    assert main(t1=t1, sd=tmp_path, sid="sub", t2=scaled_nifti(tmp_path / "other.nii.gz", slope=0.5)) == 1
    assert filecmp.cmp(t2, orig_dir / "T2raw.nii.gz", shallow=False), "the first archive is intact"
    # and the archive the error names is one the caller can pass straight back in
    assert main(t1=t1, sd=tmp_path, sid="sub", t2=orig_dir / "T2raw.nii.gz") == 0


def test_an_mgz_t2_needs_only_one_file(tmp_path):
    """Its archival copy already sits at the name the tools read, so there is nothing to convert."""
    t1 = scaled_nifti(tmp_path / "t1.nii.gz")
    t2 = mgh(tmp_path / "t2.mgz")

    assert main(t1=t1, sd=tmp_path, sid="sub", t2=t2) == 0

    t2raw = tmp_path / "sub" / "mri" / T2_RAWAVG_PATH
    assert t2raw == tmp_path / "sub" / "mri" / "orig" / "T2raw.mgz"
    assert not t2raw.is_symlink(), "it is the copy, not a link to one"
    assert filecmp.cmp(t2, t2raw, shallow=False)


def test_a_missing_input_is_reported(tmp_path):
    """And reported before anything is written, so a typo does not leave a half-built directory."""
    assert main(t1=tmp_path / "absent.mgz", sd=tmp_path, sid="sub") == 1
    assert not (tmp_path / "sub").exists()

    t1 = mgh(tmp_path / "t1.mgz")
    assert main(t1=t1, sd=tmp_path, sid="sub", t2=tmp_path / "absent.mgz") == 1
    assert not (tmp_path / "sub").exists(), "the T1 is not copied when the T2 is missing"
