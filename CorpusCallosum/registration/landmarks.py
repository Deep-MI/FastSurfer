import nibabel as nib
import numpy as np

from CorpusCallosum.data.read_write import convert_numpy_to_json_serializable
from FastSurferCNN.utils import AffineMatrix4x4, logging

logger = logging.get_logger(__name__)


def _rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute a 3x3 rotation matrix mapping one direction onto another."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-8:
        return np.eye(3, dtype=float)
    kx, ky, kz = cross / cross_norm
    skew = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ]
    )
    angle = np.arccos(dot)
    return np.eye(3, dtype=float) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _affine_from_rotation_and_center(rotation: np.ndarray, center: np.ndarray) -> AffineMatrix4x4:
    """Construct an affine applying a rotation about a voxel-space center."""
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = rotation
    affine[:3, 3] = center - rotation @ center
    return affine


def adjust_midplane_to_landmarks(
        orig2fsavg_vox2vox: AffineMatrix4x4,
        ac_coords_orig: np.ndarray,
        pc_coords_orig: np.ndarray,
        base_middle_vox: float,
        warning_tilt_deg: float = 15.0,
) -> tuple[AffineMatrix4x4, dict[str, object]]:
    """Minimally adjust a midsagittal transform to contain supplied AC/PC points."""
    ac_orig = np.asarray(ac_coords_orig, dtype=float)
    pc_orig = np.asarray(pc_coords_orig, dtype=float)
    if ac_orig.shape != (3,) or pc_orig.shape != (3,):
        raise ValueError("AC and PC coordinates must each contain exactly three values.")
    if not np.all(np.isfinite((ac_orig, pc_orig))):
        raise ValueError("AC and PC coordinates must be finite.")

    ac_fsavg, pc_fsavg = nib.affines.apply_affine(orig2fsavg_vox2vox, (ac_orig, pc_orig))
    acpc_direction = pc_fsavg - ac_fsavg
    acpc_length = float(np.linalg.norm(acpc_direction))
    if acpc_length < 1e-6:
        raise ValueError("AC and PC coordinates must be distinct.")
    acpc_unit = acpc_direction / acpc_length

    initial_normal = np.array([1.0, 0.0, 0.0], dtype=float)
    adjusted_normal = initial_normal - np.dot(initial_normal, acpc_unit) * acpc_unit
    adjusted_normal_norm = float(np.linalg.norm(adjusted_normal))
    if adjusted_normal_norm < 1e-6:
        raise ValueError(
            "The AC-PC line is parallel to the midsagittal plane normal; "
            "a stable sagittal plane containing both points cannot be constructed."
        )
    adjusted_normal /= adjusted_normal_norm
    if np.dot(adjusted_normal, initial_normal) < 0:
        adjusted_normal *= -1

    tilt_deg = float(
        np.degrees(np.arccos(np.clip(np.dot(adjusted_normal, initial_normal), -1.0, 1.0)))
    )
    if tilt_deg > warning_tilt_deg:
        logger.warning(
            "Supplied AC/PC points require a large midsagittal plane adjustment (%.2f degrees). "
            "Check that the coordinates are in orig.mgz voxel space.",
            tilt_deg,
        )

    midpoint = (ac_fsavg + pc_fsavg) / 2.0
    rotation = _rotation_from_vectors(adjusted_normal, initial_normal)
    rotate_affine = _affine_from_rotation_and_center(rotation, midpoint)
    rotated_midpoint = nib.affines.apply_affine(rotate_affine, midpoint)
    translate_affine = np.eye(4, dtype=float)
    translate_affine[0, 3] = base_middle_vox - rotated_midpoint[0]
    update_affine = translate_affine @ rotate_affine
    updated_vox2vox = update_affine @ orig2fsavg_vox2vox

    ac_adjusted, pc_adjusted = nib.affines.apply_affine(updated_vox2vox, (ac_orig, pc_orig))
    residuals = [float(ac_adjusted[0] - base_middle_vox), float(pc_adjusted[0] - base_middle_vox)]
    diagnostics: dict[str, object] = {
        "landmark_source": "supplied",
        "ac_coords_orig_vox": ac_orig.tolist(),
        "pc_coords_orig_vox": pc_orig.tolist(),
        "plane_tilt_deg": tilt_deg,
        "off_plane_residuals_vox": residuals,
        "update_affine": convert_numpy_to_json_serializable(update_affine),
    }
    return updated_vox2vox, diagnostics
