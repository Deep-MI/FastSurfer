import numpy as np
from nibabel import aff2axcodes

from FastSurferCNN.utils import AffineMatrix4x4
from FastSurferCNN.utils.arg_types import OrientationType, StrictOrientationType


def affine2orientation(affine: AffineMatrix4x4) -> OrientationType:
    """Generates the orientation type string from an affine matrix."""
    orientation: StrictOrientationType = "".join(aff2axcodes(affine, ("LR", "PA", "IS")))
    # make sure the affine is normalized for vox_sizes not 1
    norm_affine = affine[:3, :3] / np.linalg.norm(affine[:3, :3], keepdims=True, axis=0)
    if np.allclose(np.sum([np.isclose(np.abs(norm_affine), i) for i in (0, 1.)], axis=0), 1):
        return orientation
    else:
        return "soft " + orientation
