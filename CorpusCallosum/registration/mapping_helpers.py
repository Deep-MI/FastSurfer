import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform


def make_midplane_affine(orig_affine, slices_to_analyze=1, offset=4):
    """
    Creates an affine transformation matrix for midplane slices.

    Args:
        orig_affine: Original image affine matrix
        slices_to_analyze: Number of slices to analyze around midplane (default=1)
        offset: Additional offset in x direction (default=4)

    Returns:
        seg_affine: Affine matrix for midplane slices
    """
    # Create translation matrix to center on midplane
    orig_to_seg = np.eye(4)
    orig_to_seg[0, 3] = -256 // 2 + slices_to_analyze // 2 + offset

    # Combine with original affine
    seg_affine = orig_affine @ np.linalg.inv(orig_to_seg)

    return seg_affine


def correct_nodding(ac_pt, pc_pt):
    """
    Calculates rotation matrix to correct for head nodding based on AC-PC line orientation.

    Args:
        ac_pt: Coordinates of the anterior commissure point
        pc_pt: Coordinates of the posterior commissure point

    Returns:
        rotation_matrix: 3x3 rotation matrix to align AC-PC line with posterior direction
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

    # plot vector ac_pc_vec and posterior_vector
    # fig, ax = plt.subplots()
    # ax.quiver(0, 0, ac_pc_vec[0], ac_pc_vec[1], color='red', label='ac_pc_vec')
    # ax.quiver(0, 0, posterior_vector[0], posterior_vector[1], color='blue', label='posterior_vector')
    # ax.legend()
    # plt.show()

    return rotation_matrix


def apply_transform_to_pt(pts, T, inv=False):
    """
    Applies an homoegenous 4x4 transformation matrix to a point.

    Args:
        pts: Point coordinates to transform
        T: Transformation matrix
        inv: If True, applies inverse of transformation (default=False)

    Returns:
        Transformed point coordinates
    """
    if inv:
        T = T.copy()
        T = np.linalg.inv(T)

    if pts.ndim == 1:
        return (T @ np.hstack((pts, 1)))[:3]
    else:
        return (T @ np.concatenate([pts, np.ones((1, pts.shape[1]))]))[:3]


def get_mapping_to_standard_space(
    orig, ac_coords_3d, pc_coords_3d, orig_fsaverage_vox2vox, output_dir
):
    """
    Maps an image to standard space using AC-PC alignment.

    Args:
        orig: Original image
        ac_coords_3d: 3D coordinates of anterior commissure
        pc_coords_3d: 3D coordinates of posterior commissure
        orig_fsaverage_vox2vox: Original to fsaverage space transformation matrix
        output_dir: Directory for output files

    Returns:
        tuple: (transformation matrix, AC coords standardized, PC coords standardized,
               AC coords original, PC coords original)
    """
    image_center = np.array(orig.shape) / 2

    # correct nodding
    nod_correct_2d = correct_nodding(ac_coords_3d[1:3], pc_coords_3d[1:3])

    # convert 2D nodding correction to 3D transformation matrix
    nod_correct_3d = np.eye(4)
    nod_correct_3d[1:3, 1:3] = nod_correct_2d[:2, :2]  # Copy rotation part to y,z axes
    nod_correct_3d[1:3, 3] = nod_correct_2d[
        :2, 2
    ]  # Copy translation part to y,z axes (usually no translation)

    ac_coords_after_nodding = apply_transform_to_pt(
        ac_coords_3d, nod_correct_3d, inv=False
    )
    pc_coords_after_nodding = apply_transform_to_pt(
        pc_coords_3d, nod_correct_3d, inv=False
    )

    ac_to_center_translation = np.eye(4)
    ac_to_center_translation[0, 3] = image_center[0] - ac_coords_after_nodding[0]
    ac_to_center_translation[1, 3] = image_center[1] - ac_coords_after_nodding[1]
    ac_to_center_translation[2, 3] = image_center[2] - ac_coords_after_nodding[2]

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


def apply_transform_and_map_volume(
    volume, transform, affine, header, output_path=None, order=3, output_size=None
):
    """
    Applies transformation to a volume and saves the result.

    Args:
        volume: Input volume data
        transform: Transformation matrix to apply
        affine: Affine matrix for the output image
        header: Header for the output image
        output_path: Path to save transformed volume

    Returns:
        transformed: Transformed volume data
    """

    if output_size is None:
        output_size = np.array(volume.shape)
    transformed = affine_transform(
        volume.astype(np.float32),
        np.linalg.inv(transform),
        output_shape=output_size,
        order=order,
    )
    if output_path is not None:
        nib.save(nib.MGHImage(transformed, affine, header), output_path)
    return transformed


def make_affine(simpleITKImage):
    """
    Creates an affine transformation matrix from a SimpleITK image.

    Args:
        simpleITKImage: Input SimpleITK image

    Returns:
        affine: 4x4 affine transformation matrix in RAS coordinates
    """
    # get affine transform in LPS
    c = [
        simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
        for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0))
    ]
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
    outputs_soft,
    orig_fsaverage_vox2vox,
    orig,
    slices_to_analyze,
    orig_space_segmentation_path=None,
    fsaverage_middle=128,
):
    """
    Maps soft labels back to original image space and applies post-processing.

    # TODO: this could by padding after the transform

    Args:
        outputs_soft: Soft label predictions
        orig_fsaverage_vox2vox: Original to fsaverage space transformation
        orig: Original image
        slices_to_analyze: Number of slices to analyze

    Returns:
        segmentation_orig_space: Final segmentation in original image space
    """
    # map softlabels to original image
    softlabels_transformed = []
    for i in range(outputs_soft.shape[-1]):

        # pad to original image size
        outputs_soft_padded = np.zeros(orig.shape)
        outputs_soft_padded[
            fsaverage_middle
            - slices_to_analyze // 2 : fsaverage_middle
            + slices_to_analyze // 2
            + 1
        ] = outputs_soft[..., i]

        s = affine_transform(
            outputs_soft_padded,
            orig_fsaverage_vox2vox,
            output_shape=orig.shape,
            order=1,
            cval=1.0 if i == 0 else 0.0,
        )
        softlabels_transformed.append(s)

    softlabels_orig_space = np.stack(softlabels_transformed, axis=-1)

    # apply softmax to softlabels_orig_space
    softlabels_orig_space = np.exp(softlabels_orig_space) / np.sum(
        np.exp(softlabels_orig_space), axis=-1, keepdims=True
    )

    segmentation_orig_space = np.argmax(softlabels_orig_space, axis=-1)
    segmentation_orig_space = np.where(
        segmentation_orig_space == 1, 192, segmentation_orig_space
    )
    segmentation_orig_space = np.where(
        segmentation_orig_space == 2, 250, segmentation_orig_space
    )

    if orig_space_segmentation_path is not None:
        nib.save(
            nib.MGHImage(segmentation_orig_space, orig.affine, orig.header),
            orig_space_segmentation_path,
        )

    return segmentation_orig_space


def interpolate_midplane(orig, orig_fsaverage_vox2vox, slices_to_analyze):
    """
    Interpolates image data at the midplane using a grid of points.

    Args:
        orig: Original image
        orig_fsaverage_vox2vox: Original to fsaverage space transformation
        slices_to_analyze: Number of slices to analyze

    Returns:
        transformed: Interpolated image data at midplane
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
        orig.get_fdata(),
        grid_orig[0:3, :],  # use only x,y,z coordinates (drop homogeneous coordinate)
        order=2,
        mode="constant",
        cval=0,
        prefilter=True,
    ).reshape(len(x_coords), len(y_coords), len(z_coords))

    return transformed
