import numpy as np
import pytest
from nibabel.orientations import aff2axcodes

from FastSurferCNN.utils import AffineMatrix4x4
from FastSurferCNN.utils.arg_types import OrientationType, StrictOrientationType


@pytest.fixture(scope="session", params=["soft LIA", "soft ARS"])
def soft_orientation(request) -> OrientationType:
    return request.param


@pytest.fixture(scope="session", params=["soft LIA", "soft ARS", "PIL"])
def orientation(request) -> OrientationType:
    return request.param


@pytest.fixture(scope="session", params=["LIA", "ARS", "PIL"])
def strict_orientation(request) -> StrictOrientationType:
    return request.param


@pytest.fixture(scope="session", params=[0.8, 1.0])
def vox_size(request) -> float:
    return request.param


@pytest.fixture(scope="session", params=[8, 15]) # [128, 256])
def img_size(request) -> float:
    return request.param


@pytest.fixture(scope="session")
def random_affine(img_size: int, vox_size: float) -> AffineMatrix4x4:
    from scipy.spatial.transform import Rotation
    affine = np.eye(4, dtype=np.float64)
    vec = np.random.randn(3)
    rotvec = vec / np.linalg.norm(vec, axis=0) * np.random.rand(1) * np.pi
    affine[:3, :3] = Rotation.from_rotvec(rotvec, False).as_matrix() * vox_size
    try:
        L_axis = aff2axcodes(affine, ("LR", "PA", "IS")).index("L")
        affine[:3, L_axis] *= -1
    except ValueError:
        pass
    affine[:3, 3] = (np.random.rand(3) - 0.5) * img_size
    return affine
