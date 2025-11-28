from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from numpy import typing as npt
from scipy.ndimage import affine_transform

from CorpusCallosum.data.constants import CC_LABEL, FORNIX_LABEL
from FastSurferCNN.utils import logging
from FastSurferCNN.utils.parallel import thread_executor

logger = logging.get_logger(__name__)


def make_midplane_affine(
        orig_affine: npt.NDArray[float],
        slices_to_analyze: int = 1,
        offset: int = 4,
    ) -> npt.NDArray[float]:
    """Create affine transformation matrix for midplane slices.

    Parameters
    ----------
    orig_affine : np.ndarray
        Original image affine matrix (4x4).
    slices_to_analyze : int, default=1
        Number of slices to analyze around midplane.
    offset : int, default=4
        Additional offset in x direction.

    Returns
    -------
    np.ndarray
        4x4 affine matrix for midplane slices.
    """
    # Create translation matrix to center on midplane
    orig_to_seg = np.eye(4)
    orig_to_seg[0, 3] = -256 // 2 + slices_to_analyze // 2 + offset

    # Combine with original affine
    seg_affine = orig_affine @ np.linalg.inv(orig_to_seg)

    return seg_affine


def correct_nodding(ac_pt: npt.NDArray[float], pc_pt: npt.NDArray[float]) -> npt.NDArray[float]:
    """Calculate rotation matrix to correct head nodding.

    Calculates rotation matrix to align AC-PC line with posterior direction,
    correcting for head nodding based on AC-PC line orientation.

    Parameters
    ----------
    ac_pt : np.ndarray
        Coordinates of the anterior commissure point.
    pc_pt : np.ndarray
        Coordinates of the posterior commissure point.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix to align AC-PC line with posterior direction.
    """
    ac_pc_vec = pc_pt - ac_pt
    ac_pc_dist = np.linalg.norm(ac_pc_vec)

    posterior_vector = np.array([0, -ac_pc_dist])

    # get angle between ac_pc_vec and posterior_vector
    dot_product = np.dot(ac_pc_vec, posterior_vector)
    norms_product = np.linalg.norm(ac_pc_vec) * np.linalg.norm(posterior_vector)
    theta = np.arccos(dot_product / norms_product)

    # Determine the sign of the angle using cross product
    cross_product = np.cross(ac_pc_vec, posterior_vector)
    if cross_product < 0:
        theta = -theta

    # create rotation matrix for theta
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    return rotation_matrix


def apply_transform_to_pt(pts: npt.NDArray[float], T: npt.NDArray[float], inv: bool = False) -> npt.NDArray[float]:
    """Apply homogeneous transformation matrix to points.

    Parameters
    ----------
    pts : np.ndarray
        Point coordinates to transform, shape (3,) or (3, N).
    T : np.ndarray
        4x4 homogeneous transformation matrix.
    inv : bool, default=False
        If True, applies inverse of transformation.

    Returns
    -------
    np.ndarray
        Transformed point coordinates, shape (3,) or (3, N).
    """
    if inv:
        T = np.linalg.inv(T)

    if pts.ndim == 1:
        return (T @ np.hstack((pts, 1)))[:3]
    else:
        return (T @ np.concatenate([pts, np.ones((1, pts.shape[1]))]))[:3]


def calc_mapping_to_standard_space(
    orig: "nib.Nifti1Image", 
    ac_coords_3d: npt.NDArray[float], 
    pc_coords_3d: npt.NDArray[float], 
    orig_fsaverage_vox2vox: npt.NDArray[float],
) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    """Get transformations to map image to standard space.

    Parameters
    ----------
    orig : nib.Nifti1Image
        Original image.
    ac_coords_3d : np.ndarray
        AC coordinates in 3D space.
    pc_coords_3d : np.ndarray
        PC coordinates in 3D space.
    orig_fsaverage_vox2vox : np.ndarray
        Transformation matrix from original to fsaverage space.

    Returns
    -------
    upright_volume : np.ndarray
        Upright transformed volume.
    standardized_volume : np.ndarray
        Volume in standard space.
    ac_coords_standardized : np.ndarray
        AC coordinates in standard space.
    pc_coords_standardized : np.ndarray
        PC coordinates in standard space.
    standardized_affine : np.ndarray
        Affine matrix for standard space.
    """
    image_center = np.array(orig.shape) / 2

    # correct nodding
    nod_correct_2d = correct_nodding(ac_coords_3d[1:3], pc_coords_3d[1:3])

    # convert 2D nodding correction to 3D transformation matrix
    nod_correct_3d = np.eye(4)
    nod_correct_3d[1:3, 1:3] = nod_correct_2d[:2, :2]  # Copy rotation part to y,z axes
    # Copy translation part to y,z axes (usually no translation)
    nod_correct_3d[1:3, 3] = nod_correct_2d[:2, 2]

    ac_coords_after_nodding = apply_transform_to_pt(
        ac_coords_3d, nod_correct_3d, inv=False
    )
    pc_coords_after_nodding = apply_transform_to_pt(
        pc_coords_3d, nod_correct_3d, inv=False
    )

    ac_to_center_translation = np.eye(4)
    ac_to_center_translation[:3, 3] = image_center - ac_coords_after_nodding

    # correct nodding
    ac_coords_standardized = apply_transform_to_pt(
        ac_coords_after_nodding, ac_to_center_translation, inv=False
    )
    pc_coords_standardized = apply_transform_to_pt(
        pc_coords_after_nodding, ac_to_center_translation, inv=False
    )

    standardized_to_orig_vox2vox = (
        np.linalg.inv(orig_fsaverage_vox2vox)
        @ np.linalg.inv(nod_correct_3d)
        @ np.linalg.inv(ac_to_center_translation)
    )

    # calculate ac & pc in space of mri input image
    ac_coords_orig = apply_transform_to_pt(
        ac_coords_standardized, standardized_to_orig_vox2vox, inv=False
    )
    pc_coords_orig = apply_transform_to_pt(
        pc_coords_standardized, standardized_to_orig_vox2vox, inv=False
    )

    return (
        standardized_to_orig_vox2vox,
        ac_coords_standardized,
        pc_coords_standardized,
        ac_coords_orig,
        pc_coords_orig,
    )


def apply_transform_to_volume(
    orig_image: nib.analyze.SpatialImage,
    transform: npt.NDArray[float],
    affine: npt.NDArray[float],
    header: nib.freesurfer.mghformat.MGHHeader | None = None,
    output_path: str | Path | None = None,
    output_size: np.ndarray | None = None,
    order: int = 1
) -> npt.NDArray[float]:
    """Apply transformation to a volume and save the result.

    Parameters
    ----------
    orig_image : nibabel.analyze.SpatialImage
        Input volume.
    transform : np.ndarray
        Transformation matrix to apply.
    affine : np.ndarray
        Affine matrix for the output image.
    header : nib.freesurfer.mghformat.MGHHeader, optional
        Header for the output image, if None will default to orig_image header.
    output_path : str or Path, optional
        Path to save transformed volume.
    output_size : np.ndarray, optional
        Size of output volume, uses input size by default (`None`).
    order : int, default=1
        Order of interpolation.

    Returns
    -------
    npt.NDArray[float]
        Transformed volume data.

    Notes
    -----
    Uses scipy.ndimage.affine_transform for the transformation.
    If output_path is provided, saves the result as a MGH file.
    """
    if output_size is None:
        output_size = np.array(orig_image.shape)
    if header is None:
        header = orig_image.header
    transformed = affine_transform(
        orig_image.get_data(),
        np.linalg.inv(transform),
        output_shape=output_size,
        order=order,
    )
    if output_path is not None:
        logger.info(f"Saving transformed volume to {output_path}")
        nib.save(nib.MGHImage(transformed.astype(orig_image.get_data_dtype()), affine, header), output_path)
    return transformed


def make_affine(simpleITKImage: sitk.Image) -> npt.NDArray[float]:
    """Create an affine transformation matrix from a SimpleITK image.

    Parameters
    ----------
    simpleITKImage : sitk.Image
        Input SimpleITK image.

    Returns
    -------
    np.ndarray
        4x4 affine transformation matrix in RAS coordinates.

    Notes
    -----
    The function:
    1. Gets affine transform in LPS coordinates
    2. Converts to RAS coordinates to match nibabel
    3. Returns the final 4x4 transformation matrix
    """
    # get affine transform in LPS
    c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p) for p in np.eye(4)[:, :3]]
    c = np.array(c)
    affine = np.concatenate(
        [np.concatenate([c[0:3] - c[3:], c[3:]], axis=0), [[0.0], [0.0], [0.0], [1.0]]],
        axis=1,
    )
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1.0, -1.0, 1.0, 1.0]), affine)
    return affine


def map_softlabels_to_orig(
    cc_fn_softlabels: npt.NDArray[float],
    orig_fsaverage_vox2vox: npt.NDArray[float],
    orig: nib.analyze.SpatialImage,
    orig_space_segmentation_path: str | Path | None = None,
    fsaverage_middle: int = 128,
    cc_subseg_midslice: npt.NDArray[int] | None = None
) -> npt.NDArray[int]:
    """Map soft labels back to original image space and apply post-processing.

    Parameters
    ----------
    cc_fn_softlabels : np.ndarray
        Soft label predictions.
    orig_fsaverage_vox2vox : np.ndarray
        Original to fsaverage space transformation.
    orig : nibabel.analyze.SpatialImage
        Original image.
    orig_space_segmentation_path : str or Path, optional
        Path to save segmentation in original space.
    fsaverage_middle : int, default=128
        Middle slice index in fsaverage space.
    cc_subseg_midslice : npt.NDArray[int], optional
        Mask for subdividing regions.

    Returns
    -------
    npt.NDArray[int]
        Final segmentation in original image space.

    Notes
    -----
    The function:
    1. Transforms background, cc, and fornix label channels separately.
    2. Transform CC subsegmentation from midslice to orig and paint into segmentation if `cc_subseg_midslice` is passed.
    4. Saves result to `orig_space_segmentation_path` if passed.
    """
    slices_to_analyze = cc_fn_softlabels.shape[0]
    # map softlabels to original image
    slab2fsaverage_vox2vox = np.eye(4)
    slab2fsaverage_vox2vox[0, 3] = -(fsaverage_middle - slices_to_analyze // 2)
    slab2orig_vox2vox = orig_fsaverage_vox2vox @ slab2fsaverage_vox2vox

    def _map_softlabel_to_orig(i: int, data: np.ndarray) -> np.ndarray:
        return affine_transform(data, slab2orig_vox2vox, output_shape=orig.shape, order=1, cval=float(i == 0))

    _softlabels = np.moveaxis(cc_fn_softlabels, -1, 0)
    softlabels_transformed = thread_executor().map(_map_softlabel_to_orig, *zip(*enumerate(_softlabels), strict=True))

    softlabels_orig_space = np.stack(list(softlabels_transformed), axis=-1)
    seg_orig_space = np.argmax(softlabels_orig_space, axis=-1)
    # map to freesurfer labels
    seg_lut = np.asarray([0, CC_LABEL, FORNIX_LABEL])
    seg_orig_space = seg_lut[seg_orig_space]

    if cc_subseg_midslice is not None:
        # map subdivision mask to orig space
        midslice2fsaverage_vox2vox = np.eye(4)
        midslice2fsaverage_vox2vox[0, 3] = -fsaverage_middle
        cc_subseg_orig_space = affine_transform(
            cc_subseg_midslice[None],
            orig_fsaverage_vox2vox,
            output_shape=orig.shape,
            order=0,
            mode="nearest",
        )

        seg_orig_space = np.where(seg_orig_space == CC_LABEL, cc_subseg_orig_space, seg_orig_space)

    if orig_space_segmentation_path is not None:
        logger.info(f"Saving segmentation in original space to {orig_space_segmentation_path}")
        nib.save(
            nib.MGHImage(seg_orig_space, orig.affine, orig.header),
            orig_space_segmentation_path,
        )

    return seg_orig_space


def interpolate_midplane(
        orig: nib.Nifti1Image, 
        orig_fsaverage_vox2vox: np.ndarray, 
        slices_to_analyze: int) -> np.ndarray:
    """Interpolates image data at the midplane using a grid of points.

    Parameters
    ----------
    orig : nib.Nifti1Image
        Original image.
    orig_fsaverage_vox2vox : np.ndarray
        Original to fsaverage space transformation matrix.
    slices_to_analyze : int
        Number of slices to analyze around midplane.

    Returns
    -------
    np.ndarray
        Interpolated image data at midplane.
    """

    # slice_thickness = 9+slices_to_analyze-1
    # make grid of 9 slices in the fsaverage middle 
    # (cube from 123.5,0.5,0.5 to 132.5,255.5,255.5 (incudling end points, 1mm spacing))
    x_coords = np.linspace(
        124 - slices_to_analyze // 2,
        132 + slices_to_analyze // 2,
        9 + (slices_to_analyze - 1),
        endpoint=True,
    )  # 9 points from 123.5 to 132.5
    y_coords = np.linspace(
        0, orig.shape[1] - 1, orig.shape[1], endpoint=True
    )  # 255 points from 0.5 to 255.5
    z_coords = np.linspace(
        0, orig.shape[2] - 1, orig.shape[2], endpoint=True
    )  # 255 points from 0.5 to 255.5
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    # Stack coordinates and add homogeneous coordinate
    grid_fsaverage = np.stack([X.ravel(), Y.ravel(), Z.ravel(), np.ones(X.size)])

    # move grid to orig space by applying transform
    grid_orig = np.linalg.inv(orig_fsaverage_vox2vox) @ grid_fsaverage

    # interpolate grid on orig image
    from scipy.ndimage import map_coordinates

    transformed = map_coordinates(
        np.asarray(orig.dataobj),
        grid_orig[0:3, :],  # use only x,y,z coordinates (drop homogeneous coordinate)
        order=2,
        mode="constant",
        cval=0,
        prefilter=True,
    ).reshape(len(x_coords), len(y_coords), len(z_coords))

    return transformed
