import copy
from logging import getLogger

import nibabel as nib
import numpy as np
import pytest
from nibabel.orientations import aff2axcodes
from numpy import typing as npt
from pytest import approx
from scipy.ndimage import affine_transform

from FastSurferCNN.data_loader.conform import (
    IntVector3d,
    Reorientation,
    _translation_to_fix_center,
    apply_vox2vox,
    does_vox2vox_rot_require_interpolation,
    ornt2vox2vox,
)
from FastSurferCNN.utils import AffineMatrix4x4, Image3d, Shape3d, Vector3d
from FastSurferCNN.utils.arg_types import OrientationType, StrictOrientationType

logger = getLogger(__name__)
SQRT1_2 = np.sqrt(0.5)
SQRT3_4 = np.sqrt(0.75)


@pytest.fixture(scope="module", params=[(5, 2, 4), (2, 7, 3)])
def in_shape(request) -> Shape3d:
    return request.param


@pytest.fixture(scope="module", params=[(4, 6, 8), (4, 3, 4)])
def out_shape(request) -> Shape3d:
    return request.param


@pytest.fixture(scope="class")
def out_shape_reordered_strict(strict_orientation: StrictOrientationType, out_shape: Shape3d) -> Shape3d:
    """The shape of the output image after applying vox2vox from strict_orientation."""
    # reorder is <strict_orientation> => RAS
    reorder = nib.orientations.axcodes2ornt(strict_orientation, ("LR", "PA", "IS"))[:, 0].astype(np.int16)
    return tuple(np.asarray(out_shape)[reorder].tolist())


@pytest.fixture(params=[np.ones((3,)), np.array([0.8, 0.9, 1])])
def vox_size_vector(request) -> Vector3d:
    return request.param


def test_translation_fix_center(
        strict_orientation: StrictOrientationType,
        vox_size_vector: Vector3d,
        in_shape: Shape3d,
        out_shape: Shape3d,
):
    """Tests whether the translation fix for a vox2vox transformation matrix is correct"""
    _in_shape = np.asarray(in_shape)
    _out_shape = np.asarray(out_shape)

    # ornt is <strict> to RAS, so strict in, ras out
    ornt = nib.orientations.axcodes2ornt(strict_orientation, ("LR", "PA", "IS")).astype(np.int16)
    # fill diagonal with signed vox sizes, i.e. include flip
    vox2vox_strict2ras = np.eye(4) * np.append(vox_size_vector * ornt[ornt[:, 0], 1], 1)
    # reorder columns
    vox2vox_ras2strict = vox2vox_strict2ras[:, np.append(ornt[:, 0], -1)]
    # expected vox center
    expected = (_out_shape - 1) / 2
    # Note, translation_to_fix_center wants the shape in input order space!
    vox2vox_ras2strict[:3, 3] = _translation_to_fix_center(vox2vox_ras2strict, _in_shape, _out_shape[ornt[:, 0]])
    # actual center
    actual = (np.linalg.inv(vox2vox_ras2strict) @ np.append((_in_shape - 1) / 2, 1))[:3]
    assert actual == approx(expected), "Inconsistent image centers to fix center from ornt2vox2vox!"


@pytest.mark.parametrize(argnames=["shape"], argvalues=[[(1,) * 2]])
def test_ornt2vox2vox_valueerror_shape(shape: tuple[int]):
    """Test whether ornt2vox2vox raises the correct ValueError"""
    with pytest.raises(ValueError, match="length of shape"):
        ornt2vox2vox(np.transpose([np.arange(3), np.ones((3,))]), shape=shape)


@pytest.mark.parametrize(argnames=["scale"], argvalues=[[np.asarray([0.8, 0.9])], ["10"]])
def test_ornt2vox2vox_valueerror_scale(scale: float | list[float]):
    """Test whether ornt2vox2vox raises the correct ValueError"""
    with pytest.raises(ValueError, match="scale"):
        ornt2vox2vox(np.transpose([np.arange(3), np.ones((3,))]), shape=(1,) * 3, scale=scale)


@pytest.mark.parametrize(argnames="shape", argvalues=[(1, 1, 1)])
def test_ornt2vox2vox_shape(shape):
    """Test whether ornt2vox2vox returns the correct output shape."""
    actual = ornt2vox2vox(np.transpose([np.arange(3), np.ones((3,))]), shape=shape).shape
    expected = (4,) * 2
    assert actual == expected, "The shape of ornt2vox2vox was incorrect!"


def test_ornt2vox2vox_axcode(strict_orientation: StrictOrientationType):
    """Test whether ornt2vox2vox returns the an affine of the correct axcode."""
    vox2vox_strict2ras = ornt2vox2vox(nib.orientations.axcodes2ornt(strict_orientation, ("LR", "PA", "IS")), (1,) * 3)
    actual = "".join(aff2axcodes(vox2vox_strict2ras, ("LR", "PA", "IS")))
    expected = strict_orientation
    assert actual == expected, "ornt2vox2vox did not return a vox2vox of the correct orientation."


@pytest.mark.parametrize(argnames=["axcode", "translation"], argvalues=[["ALS", [1, 0, 0]], ["PIR", [0, 1, 1]]])
def test_ornt2vox2vox_translation(axcode: StrictOrientationType, translation: Vector3d, img_size: int, vox_size: float):
    """Test whether the translation of ornt2vox2vox is correct."""
    # axcode => ras
    ornt = nib.orientations.axcodes2ornt(axcode)
    out_img_size = int(np.ceil(img_size / vox_size).item())
    # the correct translation vector should consist of the combination of input and output shape.
    # if the image width stays the same, but the vox_size changes, the corners move (i.e. delta is not zero)
    # we are considering vox_size as the vox_size of out and vox_size of in as 1.
    center_out_minus_in = (out_img_size - 1) * vox_size / 2 - (img_size - 1) / 2  # >= 0!
    # expected is the correction in ras (out) coordinates
    expected = np.asarray(translation) * (img_size - 1 + 2 * center_out_minus_in) - center_out_minus_in
    actual = ornt2vox2vox(ornt, shape=(img_size,) * 3, scale=vox_size)[:3, 3]
    assert actual == approx(expected), "Translation component of the vox2vox from ornt2vox2vox did not match!"


@pytest.mark.parametrize(argnames=["axcode", "translation"], argvalues=[["ALS", [0, 1, 0]], ["PIR", [1, 1, 0]]])
def test_ornt2vox2vox_translation_corner(
        axcode: StrictOrientationType,
        translation: Vector3d,
        img_size: int,
        vox_size: float,
):
    """Test the affine base point corner of the translation of ornt2vox2vox is correct (-0.5)³ of in image."""
    ornt = nib.orientations.axcodes2ornt(axcode)
    # this is the size of the input image resized (float value)
    out_img_size = img_size / vox_size
    # if the image gets bigger, there is going to be an offset for the corners (out shape (int) - out_img_size (float))
    corner_offset = int(np.ceil(out_img_size).item()) - out_img_size
    corner_hom = np.append(np.ones((3,)) * -0.5, 1)
    expected = corner_hom[:3] + corner_offset / 2 + (np.asarray(translation) * out_img_size)
    actual = (np.linalg.inv(ornt2vox2vox(ornt, shape=(img_size,) * 3, scale=vox_size)) @ corner_hom)[:3]
    assert actual == approx(expected), "Translation component of the vox2vox from ornt2vox2vox did not match!"


@pytest.mark.parametrize(
    argnames=["axcode", "translation"],
    argvalues=[["ALS", [SQRT3_4, 0.5, 0]], ["PIR", [-0.5, SQRT3_4, 1]], ["PIL", [SQRT3_4 - 0.5, 0.5 + SQRT3_4, 1]]],
)
def test_ornt2vox2vox_translation2(axcode: StrictOrientationType, translation: npt.ArrayLike, img_size: int):
    """Test whether the translation of ornt2vox2vox is correct."""
    from scipy.spatial.transform import Rotation

    ornt = nib.orientations.axcodes2ornt(axcode)
    img_affine = np.pad(Rotation.from_euler("XYZ", [0, 0, 30], degrees=True).as_matrix(), ((0, 1), (0, 1)))
    img_affine[:, 3] = [0, 5, 0, 1]

    vox2vox = ornt2vox2vox(ornt, shape=(img_size,) * 3)
    actual = (img_affine @ vox2vox)[:3, 3]
    expected = np.asarray(translation) * (img_size - 1) + img_affine[:3, 3]
    assert actual == approx(expected), "Translation component of the vox2vox from ornt2vox2vox did not match!"


def test_ornt2vox2vox_data(strict_orientation: OrientationType, img_size: int):
    """Test whether ornt2vox2vox + apply_vox2vox equals scipy.ndimage.affine_transform."""
    from scipy.ndimage import affine_transform

    shape = (img_size,) * 3
    # not actually using the strict re-orientation
    ornt = nib.orientations.axcodes2ornt(strict_orientation, ("LR", "PA", "IS"))
    vox2vox = ornt2vox2vox(ornt, shape=shape)
    data = np.random.randn(*shape)
    expected = nib.orientations.apply_orientation(data, ornt)
    # affine_transform applies the inverse of the given transformation, so we need to invert vox2vox here
    actual = affine_transform(data, np.linalg.inv(vox2vox))
    assert actual == approx(expected), "affine_transform and apply_affine did not yield the same result!"


def test_ornt2vox2vox_loop(strict_orientation: OrientationType, vox_size_vector: Vector3d, in_shape: IntVector3d):
    """Test circular """
    ornt = nib.orientations.axcodes2ornt(strict_orientation, ("LR", "PA", "IS")).astype(np.int16)
    expected = strict_orientation
    affine = ornt2vox2vox(ornt, shape=in_shape, scale=vox_size_vector)
    actual = "".join(nib.orientations.aff2axcodes(affine, ("LR", "PA", "IS")))
    assert actual == expected, "Given and computed axcodes of aff2axcodes(ornt2vox2vox(axcodes2ornt())) don't agree!"


@pytest.mark.parametrize("vox_size", [0.4, np.ones((3,)), np.asarray([0.8, 0.9, 1.0])])
def test_ornt2vox2vox_vox_size(strict_orientation: OrientationType, vox_size: Vector3d | float):
    """Test whether ornt2vox2vox inserts the correct voxel sizes in the correct dimensions."""
    ornt = nib.orientations.axcodes2ornt(strict_orientation, ("LR", "PA", "IS")).astype(np.int16)
    vox2vox = ornt2vox2vox(ornt, shape=(1,) * 3, scale=vox_size)
    actual = np.linalg.norm(vox2vox, axis=0)[:3]
    expected = np.full((3,), vox_size) if isinstance(vox_size, float) else vox_size
    assert actual == approx(expected), "Voxel sizes in the vox2vox from ornt2vox2vox did not match!"


def test_affine_from_target_orientation(random_affine: AffineMatrix4x4, img_size: int, orientation: OrientationType):
    """Test whether affine_for_target_orientation works as expected."""
    to_orient = Reorientation.from_target_orientation(random_affine, orientation, (img_size,) * 3)
    #  orientation2ras <= random2ras @ orientation2random
    combined = random_affine @ to_orient.vox2vox
    actual = "".join(aff2axcodes(combined, ("lr", "pa", "is")))
    expected = orientation.lower().removeprefix("soft").lstrip("-_ ")
    assert actual == expected, f"affine_for_target_orientation did not yield a transformation to {orientation}!"


def test_Reorientation_target_shape(
        strict_orientation: StrictOrientationType,
        in_shape: Shape3d,
        out_shape: Shape3d,
        out_shape_reordered_strict: Shape3d,
):
    """Test whether the computed shape of conform.Reorientation is correct."""
    obj = Reorientation.from_target_orientation(np.eye(4), strict_orientation, in_shape, 1, out_shape)
    actual = obj.target_shape
    expected = out_shape_reordered_strict
    assert actual == approx(expected), "The target shape of Reorientation was not correct."

class ReorientationTests:
    """Tests the function conform.Reorientation."""

    @pytest.fixture(scope="class")
    def obj(self, random_affine: AffineMatrix4x4, target_orientation: OrientationType, img_size: int) -> Reorientation:
        """Test whether the axcodes of an affine transformed by conform.Reorientation are correct."""
        return Reorientation.from_target_orientation(random_affine, target_orientation, (img_size,) * 3)

    def test_source_affine(self, obj: Reorientation, random_affine: AffineMatrix4x4):
        """Test whether the source affine of conform.Reorientation is correct."""
        actual = obj.source_affine
        expected = random_affine
        assert actual == approx(expected, rel=0, abs=0), "The source affine of Reorientation was not correct."

    def test_inverse_vox2vox(self, obj: Reorientation):
        """Test whether the inverse of Reorientation is correct."""
        actual = np.linalg.inv(obj.inverse.vox2vox)
        expected = obj.vox2vox
        assert actual == approx(expected), "The inverse-vox2vox of Reorientation did not yield the original vox2vox."

    def test_inverse_source_affine(self, obj: Reorientation):
        """Test whether the reverse of Reorientation is correct."""
        actual = obj.inverse.source_affine
        expected = obj.source_affine @ obj.vox2vox
        assert actual == approx(expected), "Combining source_affines of original and inverse did not yield the vox2vox."

    def test_target_axcode(self, obj: Reorientation, target_orientation: OrientationType):
        """Test whether the axcodes of the target_affine transformed by conform.Reorientation are correct."""
        from helper_functions import affine2orientation
        
        actual = affine2orientation(obj.target_affine).lower()
        expected = target_orientation.lower()
        assert actual == expected, "Axcodes of affine after conform.Reorientation were not correct."

class TestReorientationSoft(ReorientationTests):

    @pytest.fixture(scope="class")
    def target_orientation(self, strict_orientation: StrictOrientationType) -> OrientationType:
        """Test whether the axcodes of an affine transformed by conform.Reorientation are correct."""
        return "soft " + strict_orientation

    def test_soft_transform(self, obj: Reorientation):
        """Test whether the vox2vox of conform.Reorientation is a pure reorder/flip transformation."""
        # this is target2random
        actual = obj.vox2vox[:3, :3]
        expected = np.round(obj.vox2vox[:3, :3])
        assert actual == approx(expected), "The vox2vox of Reorientation was not a pure reorder/flip transformation!"

    def test_soft_transform_snap(self, obj: Reorientation):
        """Test whether the vox2vox of conform.Reorientation is a pure reorder/flip transformation."""
        other = copy.deepcopy(obj)
        other.snap_translation_to_grid_()
        actual = obj.vox2vox[:3, 3]
        expected = np.round(obj.vox2vox[:3, 3])
        assert actual == approx(expected), "The vox2vox-translation of Reorientation did not snap to grid!"


class TestReorientationStrict(ReorientationTests):

    @pytest.fixture(scope="class")
    def target_orientation(self, strict_orientation: StrictOrientationType) -> OrientationType:
        """Test whether the axcodes of an affine transformed by conform.Reorientation are correct."""
        return strict_orientation


class TestReorientationNative(ReorientationTests):

    @pytest.fixture(scope="class")
    def target_orientation(self) -> OrientationType:
        """Test whether the axcodes of an affine transformed by conform.Reorientation are correct."""
        return "native"

    def test_target_axcode(self, obj: Reorientation, target_orientation: OrientationType):
        """Test whether the axcodes of the target_affine transformed by conform.Reorientation are correct."""
        from helper_functions import affine2orientation

        super().test_target_axcode(obj, affine2orientation(obj.source_affine))

    def test_native_transform(self, obj: Reorientation):
        """Test whether the vox2vox of conform.Reorientation is a identity transform."""
        actual = obj.vox2vox
        expected = np.eye(4)
        assert actual == approx(expected), "The vox2vox of Reorientation was not an identity!"


class TestApplyVox2vox:

    @pytest.fixture(scope="module")
    def data(self, in_shape: Shape3d) -> Image3d:
        """data to flip/reorder."""
        return np.random.randn(*in_shape)

    @pytest.fixture(scope="class")
    def obj(self, in_shape: Shape3d, out_shape: Shape3d, strict_orientation: StrictOrientationType) \
            -> Reorientation:
        r = Reorientation.from_target_orientation(np.eye(4), strict_orientation, in_shape, 1, out_shape)
        r.snap_translation_to_grid_()
        return r

    @pytest.fixture(scope="class")
    def expected_result(self, obj: Reorientation, data: Image3d, out_shape_reordered_strict: Shape3d) -> np.ndarray:
        """Test whether the vox2vox of conform.Reorientation is a pure reorder/flip transformation."""
        _vox2vox = obj.vox2vox
        if not does_vox2vox_rot_require_interpolation(_vox2vox):
            return affine_transform(data, _vox2vox, output_shape=out_shape_reordered_strict, order=1, prefilter=False)
        else:
            pytest.skip("vox2vox requires interpolation!")

    @pytest.fixture(scope="class")
    def result(self, obj: Reorientation, data: Image3d) -> np.ndarray:
        """Test whether the vox2vox of conform.Reorientation is a pure reorder/flip transformation."""
        if not does_vox2vox_rot_require_interpolation(obj.vox2vox):
            return apply_vox2vox(data, obj.vox2vox, out_shape=obj.target_shape)
        else:
            pytest.skip("vox2vox requires interpolation!")

    def test_apply_vox2vox(
            self,
            expected_result: Image3d,
            result: Image3d,
    ):
        """Check the consistency of apply_vox2vox and affine_transform."""
        # apply_vox2vox uses affine_transform, if the vox2vox requires interpolation => scale = 1
        expected = expected_result
        actual = result
        assert actual == approx(expected), \
            "apply_vox2vox and affine_transform differ for a vox2vox with no interpolation!"

    def test_apply_vox2vox_shape(
            self,
            result: Image3d,
            out_shape_reordered_strict: Shape3d,
    ):
        """Check the consistency of apply_vox2vox and affine_transform."""
        actual = result.shape
        expected = out_shape_reordered_strict
        assert actual == expected, "shape of apply_vox2vox does not match requested shape!"
