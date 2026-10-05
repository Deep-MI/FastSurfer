"""Tests for the geometry of .mgz files written through SimpleITK, as the N4 bias correction does.

``orig_nu.mgz`` and every volume FreeSurfer derives from it, the surfaces included, carry the
geometry this round trip writes, while the segmentations and ``talairach.xfm``'s source volume carry
the one of ``orig.mgz``. FreeSurfer compares the two centres to float32 precision and warns when
they differ (#855), so the round trip has to keep the input's geometry exactly.
"""

import nibabel as nib
import numpy as np
import pytest

from recon_surf.image_io import readITKimage, writeITKimage

# LIA, as conform writes it, with centres taken from real scans that are not centred on the origin
LIA = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def make_conformed(path, shape, vox_size, c_ras):
    """Write an LIA .mgz of `shape` at `vox_size` whose header stores the centre `c_ras`."""
    header = nib.freesurfer.mghformat.MGHHeader()
    header.set_data_shape(shape)
    header.set_zooms((vox_size,) * 3)
    header["Mdc"] = LIA.T
    header["Pxyz_c"] = c_ras
    data = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    nib.save(nib.MGHImage(data, None, header), path)


@pytest.mark.parametrize(
    ("shape", "vox_size", "c_ras"),
    [((256, 256, 256), 1.0, (1.433895, 33.220337, -24.406784)),
     ((320, 320, 320), 0.8, (-0.114865065, 23.43119812, -25.660776))],
    ids=["1.0mm", "0.8mm"],
)
def test_sitk_round_trip_keeps_the_geometry(tmp_path, shape, vox_size, c_ras):
    """Read and written back on the same grid, the header geometry is bit for bit the input's."""
    src, out = tmp_path / "orig.mgz", tmp_path / "orig_nu.mgz"
    make_conformed(src, shape, vox_size, c_ras)
    itk_image, header = readITKimage(str(src), return_header=True)
    writeITKimage(itk_image, str(out), header)

    before, after = nib.load(src).header, nib.load(out).header
    for key in ("Pxyz_c", "Mdc", "delta"):
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)
