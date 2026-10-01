from logging import getLogger
from typing import Literal, TypedDict, cast

import nibabel as nib
import numpy as np
import pytest
from nibabel.affines import apply_affine
from nibabel.orientations import aff2axcodes
from numpy import typing as npt
from pytest import approx

from FastSurferCNN.data_loader.conform import OrientationType, conform
from FastSurferCNN.utils import AffineMatrix4x4, Image3d, nibabelHeader, nibabelImage

logger = getLogger(__name__)

class MultiCoordImages(TypedDict):
    X: nibabelImage
    Y: nibabelImage
    Z: nibabelImage

class ConformArgs(TypedDict):
    rescale: None
    dtype: npt.DTypeLike

conform_reorient: ConformArgs = {"rescale": None, "dtype": np.float32}


def circle_data(img_size: int, radius: float, center: float) -> Image3d[np.float32]:
    """Generates a 3D image with a centered sphere of radius img_size/2."""
    data = np.mgrid[0:img_size, 0:img_size, 0:img_size].astype(np.float32) - center
    return (np.sum(data * data, axis=0) < radius * radius).astype(np.float32)


def square_data(img_size: int, radius: float, center: float) -> Image3d[np.float32]:
    """Generates a 3D image with a centered sphere of radius img_size/2."""
    data: Image3d[np.float32] = np.ones((int(radius * 2),) * 3, dtype=np.float32)
    ccenter = (data.shape[0] - 1) / 2
    pad0 = int(center - ccenter)
    pad1 = img_size - pad0 - data.shape[0]
    _pads = (pad0, pad1)
    pads = (_pads, _pads, _pads)
    data: Image3d[np.float32] = np.pad(data, pads, constant_values=0)
    return data


@pytest.fixture(scope="session")
def radius(img_size: int) -> float:
    return img_size / 2.0 - 2.0


@pytest.fixture(scope="session")
def center(img_size: int) -> float:
    return (img_size - 1) / 2.0


@pytest.fixture(scope="session")
def circle_image(random_affine: AffineMatrix4x4, img_size: int, radius: float, center: float) -> nib.Nifti1Image:
    return nib.Nifti1Image(circle_data(img_size, radius, center), random_affine)


@pytest.fixture(scope="session", params=[1.0, 0.23])
def resample_factor(request) -> float:
    return request.param


@pytest.fixture(scope="session")
def random_image(random_affine: AffineMatrix4x4, img_size: int) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.random.randn(img_size, img_size, img_size), random_affine)


@pytest.fixture(scope="session")
def empty_image(random_affine: AffineMatrix4x4, img_size: int) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.zeros((img_size,) * 3, dtype=np.uint8), random_affine)


class HeaderTests:
    def test_affine_orientation(self, affine: AffineMatrix4x4, orientation: OrientationType):
        """Tests whether a conformed image actually has the correct orientation."""
        from helper_functions import affine2orientation

        actual = affine2orientation(affine)
        if orientation.startswith("soft"):
            expected = (orientation, orientation[5:])
            assert actual in expected, "The expected orientation did not match the actual orientation."
        else:
            expected = orientation
            assert actual == expected, "The expected orientation did not match the actual orientation."

    def test_affine_vox_size(self, affine: AffineMatrix4x4, vox_size: float):
        """Tests whether a conformed image actually has the correct voxel size."""
        actual = np.linalg.norm(affine[:3, :3], axis=0)
        expected = vox_size
        assert actual == approx(expected), "The actual voxel sizes in the affine did not match the expected."

    def test_vox_size(self, header: nibabelHeader, vox_size: float):
        """Tests whether a conformed image actually has the correct voxel size."""
        actual = header.get_zooms()
        expected = np.full_like(actual, vox_size)
        assert actual == approx(expected), "The actual voxel sizes in the affine did not match the expected."


class TestConformAffine(HeaderTests):

    @pytest.fixture(scope="class")
    def image(self, empty_image: nib.Nifti1Image, orientation: OrientationType, vox_size: float) -> nibabelImage:
        return conform(empty_image, orientation=orientation, vox_size=vox_size, **conform_reorient)

    @pytest.fixture(scope="class")
    def affine(self, image: nib.MGHImage) -> AffineMatrix4x4:
        return image.affine

    @pytest.fixture(scope="class")
    def header(self, image: nib.MGHImage) -> nib.freesurfer.mghformat.MGHHeader:
        return image.header


class TestThereAndBack:

    @pytest.fixture(scope="class")
    def image(
            self,
            circle_image: nib.Nifti1Image,
            soft_orientation: OrientationType,
            img_size: int,
            vox_size: float,
            resample_factor: float,
    ) -> nibabelImage:
        """
        Conform `circle_image` to `soft_orientation` and back to the original orientation.
        """
        there = conform(
            circle_image,
            orientation=soft_orientation, vox_size=vox_size * resample_factor, img_size=64, order=1,
            **conform_reorient,
        )
        in_orientation: OrientationType = "soft " + "".join(aff2axcodes(circle_image.affine, ("LR", "PA", "IS")))
        return conform(there, orientation=in_orientation, img_size=img_size, vox_size=vox_size, **conform_reorient)

    def test_affine(self, image: nib.MGHImage, circle_image: nibabelImage) -> None:
        """
        Tests whether the affines of the original and the re-oriented images are the same.
        """
        expected = circle_image.affine
        actual = image.affine
        # currently, the translation parts of the affines differ
        assert actual == approx(expected, abs=1e-5), "The affines of original and re-reoriented images differ!"

    def test_image(
            self,
            image: nibabelImage,
            circle_image: nibabelImage,
            radius: float,
            center: float,
            soft_orientation: AffineMatrix4x4,
    ) -> None:
        """
        Tests whether the content of the original and the re-oriented images are the same in the "center circle".
        """
        expected = circle_image.get_fdata()
        actual = image.get_fdata()
        max_difference = np.max(np.abs(actual - circle_image.get_fdata()))
        logger.info(f"Difference is {max_difference} - {np.nanmax(np.abs(actual - expected))}")

        # boundaries get blurred twice, so threshold with abs 0.4, so 0-0.45 -> 0; 0.55-1.0 -> 1.0
        assert actual == approx(expected, abs=0.45), "The data differs from the re-oriented image!"

class TestReorientWorldCoords:

    @staticmethod
    def worldcoords_data(affine: AffineMatrix4x4, img_size2: int) -> np.ndarray:
        xi = np.moveaxis(np.mgrid[0:img_size2, 0:img_size2, 0:img_size2], 0, -1)
        return apply_affine(affine, xi.reshape((-1, 3)).astype(np.float64)).reshape(xi.shape).astype(np.float32)

    @pytest.fixture(scope="class")
    def img_size2(self, img_size: int) -> int:
        return img_size * 2

    @pytest.fixture(scope="class")
    def worldcoord_images(self, random_affine: AffineMatrix4x4, img_size2: int) -> MultiCoordImages:
        data = self.worldcoords_data(random_affine, img_size2)
        return MultiCoordImages(
            X=nib.Nifti1Image(data[..., 0], random_affine),
            Y=nib.Nifti1Image(data[..., 1], random_affine),
            Z=nib.Nifti1Image(data[..., 2], random_affine),
        )

    @pytest.fixture(scope="class")
    def conf_img_size(self, img_size2: int) -> int:
        # this size for the conformed image avoids extrapolation
        return np.floor(img_size2 / np.sqrt(3)).astype(np.int16).item()

    @pytest.fixture(scope="class")
    def conf_images(
            self,
            conf_img_size: int,
            worldcoord_images: MultiCoordImages,
            orientation: OrientationType,
    ) -> MultiCoordImages:
        return MultiCoordImages(
            X=conform(worldcoord_images["X"], **conform_reorient, img_size=conf_img_size, orientation=orientation),
            Y=conform(worldcoord_images["Y"], **conform_reorient, img_size=conf_img_size, orientation=orientation),
            Z=conform(worldcoord_images["Z"], **conform_reorient, img_size=conf_img_size, orientation=orientation),
        )

    @pytest.mark.parametrize(argnames=["dim_name"], argvalues=[["X"], ["Y"], ["Z"]])
    def test_reorient_worldcoords_affine(
            self,
            conf_images: MultiCoordImages,
            worldcoord_images: MultiCoordImages,
            orientation: OrientationType,
            dim_name: Literal["X", "Y", "Z"],
    ):
        """
        This test checks, whether the affines of the world coordinate images are consistent with the original affine.
        """
        from helper_functions import affine2orientation

        actual = "".join(affine2orientation(conf_images[dim_name].affine))
        if orientation.lower() == "native":
            # native means the image does not change, so the orientation is determined by the original affine
            expected = "".join(affine2orientation(worldcoord_images[dim_name].affine))
        else:
            expected = orientation
        assert actual == expected, "The conformed orientation of the world coordinate image is not correct!"


    def test_reorient_worldcoords_image(
            self,
            conf_images: MultiCoordImages,
            conf_img_size: int,
    ):
        """
        This test checks, whether the world coordinates are consistent.
        """
        from pytest import approx

        assert list(conf_images.keys()) == ["X", "Y", "Z"], "The order of coordinate directions is not X, Y, Z!"

        logger.info("Checking affines of world images:")

        worldcoords = self.worldcoords_data(conf_images["X"].affine, conf_images["X"].shape[0])

        # Mask out boundary voxels that were extrapolated
        valid = circle_data(conf_img_size, conf_img_size / 2.0 - 1.0, (conf_img_size - 1) / 2.0)> 0.5
        expected = np.where(valid[..., np.newaxis], worldcoords, 0)
        conformed_worldcoords = [cast(nibabelImage, conf_image).get_fdata() for conf_image in conf_images.values()]
        actual = np.where(valid[..., np.newaxis], np.stack(conformed_worldcoords, axis=-1), 0)
        assert actual == approx(expected, abs=0.1), "The world images are not consistent after conforming!"
