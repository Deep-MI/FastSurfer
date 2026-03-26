from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform, binary_fill_holes, distance_transform_edt
from skimage.morphology import convex_hull_image

from CorpusCallosum.data.read_write import convert_numpy_to_json_serializable
from CorpusCallosum.shape.postprocessing import offset_affine
from FastSurferCNN.utils import AffineMatrix4x4, Shape3d, logging, nibabelImage
from FastSurferCNN.utils.brainvolstats import hemi_masks_from_aseg

logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class MidplaneDebugVolumes:
    distance_diff: np.ndarray
    candidate_mask: np.ndarray
    left_mask: np.ndarray
    right_mask: np.ndarray


@dataclass(frozen=True)
class MidplaneRefinementResult:
    updated_vox2vox: AffineMatrix4x4
    center_shift_vox: float
    diagnostics: dict[str, object]
    debug_volumes: MidplaneDebugVolumes


def resample_segmentation_to_fsavg(
    seg_data: np.ndarray,
    orig2fsavg_vox2vox: AffineMatrix4x4,
    fsavg_shape: Shape3d,
) -> np.ndarray:
    """Resample a segmentation into fsaverage voxel space using nearest-neighbor interpolation."""
    return affine_transform(
        seg_data.astype(np.int32),
        np.linalg.inv(orig2fsavg_vox2vox),
        output_shape=fsavg_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.int32)


def _make_fsavg_convex_hull_mask(brain_mask: np.ndarray) -> np.ndarray:
    """Approximate the whole-brain convex hull in fsaverage voxel space."""
    yz_projection = brain_mask.any(axis=0)
    yz_hull = convex_hull_image(yz_projection)

    x_min = np.full(yz_projection.shape, brain_mask.shape[0], dtype=np.int32)
    x_max = np.full(yz_projection.shape, -1, dtype=np.int32)
    x_coords, y_coords, z_coords = np.where(brain_mask)
    np.minimum.at(x_min, (y_coords, z_coords), x_coords)
    np.maximum.at(x_max, (y_coords, z_coords), x_coords)

    valid_yz = x_max >= x_min
    hull_valid = yz_hull & valid_yz
    if not hull_valid.any():
        return brain_mask.copy()

    nearest_valid = distance_transform_edt(~valid_yz, return_distances=False, return_indices=True)
    hull_mask = np.zeros_like(brain_mask, dtype=bool)
    for y_idx, z_idx in np.argwhere(yz_hull):
        if valid_yz[y_idx, z_idx]:
            x0 = int(x_min[y_idx, z_idx])
            x1 = int(x_max[y_idx, z_idx])
        else:
            src_y = int(nearest_valid[0, y_idx, z_idx])
            src_z = int(nearest_valid[1, y_idx, z_idx])
            x0 = int(x_min[src_y, src_z])
            x1 = int(x_max[src_y, src_z])
        if x1 >= x0:
            hull_mask[x0:x1 + 1, y_idx, z_idx] = True
    return hull_mask


def _clean_partitioned_hemi_masks(seg_fsavg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create hole-free left/right hemisphere masks covering the full brain mask."""
    brain_mask = seg_fsavg > 0
    left_vote, right_vote = hemi_masks_from_aseg(seg_fsavg)
    left_core = np.logical_and(left_vote, brain_mask)
    right_core = np.logical_and(right_vote, brain_mask)

    undecided = np.logical_and(brain_mask, np.logical_not(np.logical_or(left_core, right_core)))
    left_core_dt = distance_transform_edt(np.logical_not(left_core))
    right_core_dt = distance_transform_edt(np.logical_not(right_core))
    left_mask = np.logical_or(left_core, np.logical_and(undecided, left_core_dt <= right_core_dt))
    right_mask = np.logical_and(brain_mask, np.logical_not(left_mask))

    left_mask = np.logical_and(binary_fill_holes(left_mask), brain_mask)
    right_mask = np.logical_and(binary_fill_holes(right_mask), brain_mask)

    overlap = np.logical_and(left_mask, right_mask)
    if np.any(overlap):
        left_overlap = left_core_dt <= right_core_dt
        left_mask = np.logical_or(
            np.logical_and(left_mask, np.logical_not(overlap)),
            np.logical_and(overlap, left_overlap),
        )
        right_mask = np.logical_and(brain_mask, np.logical_not(left_mask))

    remainder = np.logical_and(brain_mask, np.logical_not(np.logical_or(left_mask, right_mask)))
    if np.any(remainder):
        left_mask = np.logical_or(left_mask, np.logical_and(remainder, left_core_dt <= right_core_dt))
        right_mask = np.logical_and(brain_mask, np.logical_not(left_mask))

    return left_mask, right_mask


def _rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix mapping source onto target."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-8:
        if dot > 0:
            return np.eye(3, dtype=float)
        axis = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(source[1]) > 0.9:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        cross = np.cross(source, axis)
        cross = cross / np.linalg.norm(cross)
        cross_norm = 1.0
        dot = -1.0
    k = cross / cross_norm
    kx, ky, kz = k
    skew = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    angle = np.arccos(dot)
    return np.eye(3, dtype=float) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _affine_from_rotation_and_center(rotation: np.ndarray, center: np.ndarray) -> AffineMatrix4x4:
    """Build a 4x4 affine that rotates around a given center in voxel coordinates."""
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = rotation
    affine[:3, 3] = center - rotation @ center
    return affine


def refine_midplane_with_distance_maps(
    orig2fsavg_vox2vox: AffineMatrix4x4,
    aseg_nib: nibabelImage,
    fsavg_shape: Shape3d,
    base_middle_vox: float,
    zero_band_vox: float = 0.5,
    support_distance_vox: float = 24.0,
    fit_band_vox: float = 20.0,
    max_tilt_deg: float = 7.5,
    max_center_shift_vox: float = 8.0,
) -> MidplaneRefinementResult:
    """Fit a midsagittal plane from left/right distance-map differences inside the brain hull."""
    seg_data = np.asarray(aseg_nib.dataobj).astype(np.int32)
    empty = np.zeros(fsavg_shape, dtype=np.float32)
    empty_debug = MidplaneDebugVolumes(
        distance_diff=empty,
        candidate_mask=empty.astype(bool),
        left_mask=empty.astype(bool),
        right_mask=empty.astype(bool),
    )
    if not np.any(seg_data > 0):
        logger.warning("Distance-map midplane refinement skipped: segmentation is empty.")
        return MidplaneRefinementResult(
            updated_vox2vox=orig2fsavg_vox2vox,
            center_shift_vox=0.0,
            diagnostics={},
            debug_volumes=empty_debug,
        )

    seg_fsavg = resample_segmentation_to_fsavg(seg_data, orig2fsavg_vox2vox, fsavg_shape)
    left_mask, right_mask = _clean_partitioned_hemi_masks(seg_fsavg)
    brain_mask = seg_fsavg > 0
    hull_mask = _make_fsavg_convex_hull_mask(brain_mask)

    left_distance = distance_transform_edt(~left_mask)
    right_distance = distance_transform_edt(~right_mask)
    distance_diff = left_distance - right_distance
    candidate_mask = (
        hull_mask
        & (np.abs(distance_diff) <= zero_band_vox)
        & (np.minimum(left_distance, right_distance) <= support_distance_vox)
        & (np.abs(np.arange(fsavg_shape[0])[:, np.newaxis, np.newaxis] - base_middle_vox) <= fit_band_vox)
    )
    debug_volumes = MidplaneDebugVolumes(
        distance_diff=distance_diff.astype(np.float32),
        candidate_mask=candidate_mask,
        left_mask=left_mask,
        right_mask=right_mask,
    )
    candidate_coords = np.argwhere(candidate_mask)
    diagnostics: dict[str, object] = {
        "candidate_count": int(candidate_coords.shape[0]),
        "zero_band_vox": float(zero_band_vox),
        "support_distance_vox": float(support_distance_vox),
        "fit_band_vox": float(fit_band_vox),
        "left_mask_voxels": int(np.count_nonzero(left_mask)),
        "right_mask_voxels": int(np.count_nonzero(right_mask)),
        "brain_mask_voxels": int(np.count_nonzero(brain_mask)),
    }
    if candidate_coords.shape[0] < 200:
        logger.warning(
            "Distance-map midplane refinement skipped: too few candidate voxels (%d).",
            candidate_coords.shape[0],
        )
        diagnostics["rejected_reason"] = "too_few_candidates"
        return MidplaneRefinementResult(
            updated_vox2vox=orig2fsavg_vox2vox,
            center_shift_vox=0.0,
            diagnostics=diagnostics,
            debug_volumes=debug_volumes,
        )

    x = candidate_coords[:, 0].astype(float)
    y = candidate_coords[:, 1].astype(float)
    z = candidate_coords[:, 2].astype(float)
    design = np.column_stack([y, z, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, x, rcond=None)
    a_coef, b_coef, c_coef = [float(c) for c in coeffs]
    fitted_x = design @ coeffs
    rmse = float(np.sqrt(np.mean((fitted_x - x) ** 2)))

    plane_normal = np.array([1.0, -a_coef, -b_coef], dtype=float)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    tilt_deg = float(np.degrees(np.arccos(np.clip(np.dot(plane_normal, x_axis), -1.0, 1.0))))

    yz_center = np.asarray(fsavg_shape[1:], dtype=float) / 2.0
    plane_center = np.array([
        a_coef * yz_center[0] + b_coef * yz_center[1] + c_coef,
        yz_center[0],
        yz_center[1],
    ], dtype=float)
    center_shift_vox = float(plane_center[0] - base_middle_vox)

    diagnostics.update({
        "plane_coefficients_xyz": [1.0, -a_coef, -b_coef, -c_coef],
        "plane_center_vox": plane_center.tolist(),
        "plane_fit_rmse_vox": rmse,
        "plane_tilt_deg": tilt_deg,
        "center_shift_vox": center_shift_vox,
        "max_tilt_deg": float(max_tilt_deg),
        "max_center_shift_vox": float(max_center_shift_vox),
    })

    if abs(center_shift_vox) > max_center_shift_vox:
        logger.info(
            "Distance-map midplane refinement rejected: center shift %.2f vox exceeds %.2f.",
            center_shift_vox,
            max_center_shift_vox,
        )
        diagnostics["rejected_reason"] = "center_shift_too_large"
        return MidplaneRefinementResult(
            updated_vox2vox=orig2fsavg_vox2vox,
            center_shift_vox=0.0,
            diagnostics=diagnostics,
            debug_volumes=debug_volumes,
        )
    if tilt_deg > max_tilt_deg:
        logger.info(
            "Distance-map midplane refinement rejected: tilt %.2f deg exceeds %.2f deg.",
            tilt_deg,
            max_tilt_deg,
        )
        diagnostics["rejected_reason"] = "tilt_too_large"
        return MidplaneRefinementResult(
            updated_vox2vox=orig2fsavg_vox2vox,
            center_shift_vox=0.0,
            diagnostics=diagnostics,
            debug_volumes=debug_volumes,
        )

    rotation = _rotation_from_vectors(plane_normal, x_axis)
    rotate_affine = _affine_from_rotation_and_center(rotation, plane_center)
    translate_affine = offset_affine([base_middle_vox - plane_center[0], 0, 0])
    update_affine = translate_affine @ rotate_affine
    diagnostics["update_affine"] = convert_numpy_to_json_serializable(update_affine)
    diagnostics["rejected_reason"] = None
    logger.info(
        "Distance-map midplane refinement applied center shift %.2f vox with tilt %.2f deg (RMSE %.3f, candidates %d).",
        center_shift_vox,
        tilt_deg,
        rmse,
        candidate_coords.shape[0],
    )
    return MidplaneRefinementResult(
        updated_vox2vox=update_affine @ orig2fsavg_vox2vox,
        center_shift_vox=center_shift_vox,
        diagnostics=diagnostics,
        debug_volumes=debug_volumes,
    )
