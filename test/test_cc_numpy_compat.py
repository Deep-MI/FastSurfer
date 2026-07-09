import importlib.util
from pathlib import Path

import numpy as np


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subsegment_contour = _load_module(
    "cc_subsegment_contour",
    "CorpusCallosum/shape/subsegment_contour.py",
)
mapping_helpers = _load_module(
    "cc_mapping_helpers",
    "CorpusCallosum/utils/mapping_helpers.py",
)

transform_to_acpc_standard = subsegment_contour.transform_to_acpc_standard
correct_nodding = mapping_helpers.correct_nodding


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