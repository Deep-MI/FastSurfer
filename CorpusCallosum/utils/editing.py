from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np

from CorpusCallosum.data.constants import CC_LABEL, FORNIX_LABEL
from FastSurferCNN.utils import AffineMatrix4x4, Image3d, Shape3d


def add_file_suffix(path: str | Path, suffix: str) -> Path:
    """Insert a suffix before a file extension, including compound NIfTI extensions."""
    path = Path(path)
    extension = ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix
    stem = path.name[:-len(extension)] if extension else path.name
    return path.with_name(f"{stem}.{suffix}{extension}")


def validate_landmarks_in_image(
        ac_coords: Iterable[float] | None,
        pc_coords: Iterable[float] | None,
        image_shape: tuple[int, ...],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate optional paired landmarks in original-image voxel coordinates."""
    if (ac_coords is None) != (pc_coords is None):
        raise ValueError("AC and PC coordinates must be supplied together.")
    if ac_coords is None or pc_coords is None:
        return None, None
    ac = np.asarray(ac_coords, dtype=float)
    pc = np.asarray(pc_coords, dtype=float)
    if ac.shape != (3,) or pc.shape != (3,):
        raise ValueError("AC and PC coordinates must each contain exactly three values.")
    if not np.all(np.isfinite((ac, pc))):
        raise ValueError("AC and PC coordinates must be finite.")
    upper = np.asarray(image_shape[:3], dtype=float) - 1
    if np.any(ac < 0) or np.any(pc < 0) or np.any(ac > upper) or np.any(pc > upper):
        raise ValueError(
            f"AC and PC coordinates must lie inside orig.mgz voxel bounds [0, {upper.tolist()}]."
        )
    if np.linalg.norm(pc - ac) < 1e-6:
        raise ValueError("AC and PC coordinates must be distinct.")
    return ac, pc


def load_manual_upright_segmentation(
        automatic_path: Path | None,
        manual_path: Path,
        expected_shape: Shape3d,
        expected_affine: AffineMatrix4x4,
) -> tuple[Image3d, bool]:
    """Load an upright edit, using its fornix or falling back to the automatic result."""
    if not manual_path.is_file():
        raise ValueError(f"Manual upright segmentation {manual_path} does not exist.")

    manual_img = nib.load(manual_path)
    if manual_img.shape[:3] != expected_shape:
        raise ValueError(
            f"The manual upright segmentation has shape {manual_img.shape[:3]}, expected {expected_shape}. "
            "Use the same midplane method and supplied AC/PC coordinates as the original run."
        )
    if not np.allclose(manual_img.affine, expected_affine):
        raise ValueError(
            "The manual upright segmentation affine does not match the current midsagittal slab. "
            "Use the same midplane method and supplied AC/PC coordinates as the original run."
        )

    manual = np.asarray(manual_img.dataobj)
    unknown_labels = set(np.unique(manual).astype(int).tolist()) - {0, CC_LABEL, FORNIX_LABEL}
    if unknown_labels:
        raise ValueError(
            f"Manual upright segmentation contains unsupported labels {sorted(unknown_labels)}; "
            f"only 0, {CC_LABEL}, and {FORNIX_LABEL} are allowed."
        )
    manual_cc = manual == CC_LABEL
    if not np.any(manual_cc):
        raise ValueError("Manual upright segmentation does not contain any corpus callosum voxels (label 192).")

    manual_has_fornix = np.any(manual == FORNIX_LABEL)
    if manual_has_fornix:
        return manual.astype(np.uint8), True

    if automatic_path is None or not automatic_path.is_file():
        automatic_location = automatic_path if automatic_path is not None else "not configured"
        raise ValueError(
            f"Manual upright segmentation does not contain fornix label {FORNIX_LABEL}, and automatic upright "
            f"segmentation ({automatic_location}) is missing. Supply the fornix in the manual segmentation or run "
            "FastSurfer-CC once without edits."
        )

    automatic_img = nib.load(automatic_path)
    if automatic_img.shape[:3] != expected_shape:
        raise ValueError(
            f"The automatic upright segmentation has shape {automatic_img.shape[:3]}, expected {expected_shape}. "
            "Use the same midplane method and supplied AC/PC coordinates as the original run."
        )
    if not np.allclose(automatic_img.affine, expected_affine):
        raise ValueError(
            "The automatic upright segmentation affine does not match the current midsagittal slab. "
            "Use the same midplane method and supplied AC/PC coordinates as the original run."
        )

    automatic = np.asarray(automatic_img.dataobj)
    edited = np.zeros(expected_shape, dtype=np.uint8)
    edited[automatic == FORNIX_LABEL] = FORNIX_LABEL
    edited[manual_cc] = CC_LABEL
    return edited, False
