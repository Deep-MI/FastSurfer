"""Regression tests for the field-of-view header field of the .mgz files FastSurfer writes.

``MGHHeader`` defaults ``fov`` to 0 and a NIfTI header has no ``fov`` at all, so every .mgz written
from a NIfTI input carried ``fov=0``, while the files FreeSurfer writes carry the real extent. The
value pinned here is the one FreeSurfer computes for an MGZ, the largest of the three extents, which
``mri_info`` reports regardless of what the file stores. These tests cover both the header ``conform``
builds and one inherited from a volume of a different shape.
"""

import nibabel as nib
import numpy as np
import pytest

from FastSurferCNN.data_loader.conform import conform
from FastSurferCNN.data_loader.data_utils import as_mgh_image, save_image

CUBE_1MM = ((256, 256, 256), (1.0, 1.0, 1.0))
CUBE_08MM = ((320, 320, 320), (0.8, 0.8, 0.8))


def make_image(shape, vox_size, file_type=nib.MGHImage):
    """An empty image of `shape` at `vox_size`, centred on the origin."""
    affine = np.diag([*vox_size, 1.0])
    affine[:3, 3] = [-0.5 * n * v for n, v in zip(shape, vox_size, strict=True)]
    return file_type(np.zeros(shape, dtype=np.uint8), affine)


def max_extent(shape, vox_size):
    """The largest of the three extents, which is what FreeSurfer stores in fov."""
    return max(n * v for n, v in zip(shape, vox_size, strict=True))


@pytest.mark.parametrize("file_type", [nib.MGHImage, nib.Nifti1Image], ids=["from mgz", "from nifti"])
@pytest.mark.parametrize(("shape", "vox_size"), [CUBE_1MM, CUBE_08MM], ids=["1.0mm", "0.8mm"])
def test_conform_sets_fov(file_type, shape, vox_size):
    """conform fills in fov for an .mgz target, whichever container the source arrived in."""
    conformed = conform(
        make_image(shape, vox_size, file_type),
        vox_size=vox_size[0],
        img_size="auto",
        rescale=None,
        file_type=nib.MGHImage,
    )
    assert float(conformed.header["fov"]) == pytest.approx(max_extent(shape, vox_size))


def test_conform_to_nifti_has_no_fov():
    """A NIfTI target has no fov field at all, so conform must leave it alone."""
    conformed = conform(
        make_image(*CUBE_1MM),
        vox_size=1.0,
        img_size="auto",
        rescale=None,
        file_type=nib.Nifti1Image,
    )
    assert isinstance(conformed, nib.Nifti1Image)
    with pytest.raises(ValueError):
        _ = conformed.header["fov"]


def test_save_image_writes_fov(tmp_path):
    """The 0.8mm path: a NIfTI input conformed in its own container, then written as .mgz."""
    conformed = conform(make_image(*CUBE_08MM, nib.Nifti1Image), vox_size=0.8, img_size="auto", rescale=None)
    assert isinstance(conformed, nib.Nifti1Image), "conform keeps the container of its input"

    out_file = tmp_path / "conformed.mgz"
    save_image(conformed.header, conformed.affine, np.asarray(conformed.dataobj), out_file)
    assert float(nib.load(out_file).header["fov"]) == pytest.approx(256.0)


def test_as_mgh_image_replaces_a_stale_fov():
    """An inherited header must not keep the parent's fov when the data has another shape.

    This is the corpus callosum slab, written with the header of the full conformed volume.
    """
    parent = make_image(*CUBE_08MM)
    parent.header["fov"] = 256.0
    slab = as_mgh_image(np.zeros((7, 100, 100), dtype=np.uint8), parent.affine, parent.header)
    assert float(slab.header["fov"]) == pytest.approx(max_extent((7, 100, 100), (0.8, 0.8, 0.8)))


def test_fov_is_the_largest_extent_not_the_first():
    """The two candidate rules disagree here, and FreeSurfer's mri_info reports 120 for this .mgz."""
    img = as_mgh_image(np.zeros((40, 200, 80), dtype=np.uint8), np.diag([2.0, 0.5, 1.5, 1.0]))
    assert float(img.header["fov"]) == pytest.approx(120.0)
