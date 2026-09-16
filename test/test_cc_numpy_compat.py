import numpy as np

from CorpusCallosum.shape.subsegment_contour import transform_to_acpc_standard
from CorpusCallosum.utils.mapping_helpers import correct_nodding


def test_transform_to_acpc_standard_accepts_2d_vectors_with_numpy2():
    contour = np.array([[0.0, 1.0, 2.0], [0.0, 0.5, 0.0]])
    ac_pt = np.array([0.0, 0.0])
    pc_pt = np.array([1.0, 0.0])

    contour_acpc, ac_pt_acpc, pc_pt_acpc, rotate_back = transform_to_acpc_standard(contour, ac_pt, pc_pt)

    assert contour_acpc.shape == contour.shape
    assert np.allclose(ac_pt_acpc, [0.0, 0.0])
    assert pc_pt_acpc.shape == (2,)
    assert rotate_back(contour_acpc).shape == contour.shape


def test_acpc_rotation_matrix_accepts_2d_vectors_with_numpy2():
    rotation = correct_nodding(np.array([0.0, 0.0]), np.array([1.0, 0.0]))

    assert rotation.shape == (3, 3)