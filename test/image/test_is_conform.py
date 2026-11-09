import nibabel as nib
import numpy as np
import pytest

from FastSurferCNN.data_loader.conform import conform, is_conform, is_orientation

SHAPE = (8, 8, 8)
LIA = np.array([[-1.0, 0.0, 0.0, 4.0], [0.0, 0.0, 1.0, -4.0], [0.0, -1.0, 0.0, 4.0], [0.0, 0.0, 0.0, 1.0]])
RAS = np.array([[1.0, 0.0, 0.0, -4.0], [0.0, 1.0, 0.0, -4.0], [0.0, 0.0, 1.0, -4.0], [0.0, 0.0, 0.0, 1.0]])


def tilted(affine, degrees: float = 10.0):
    """Rotate the affine about the RAS z axis, small enough to keep its axis codes."""
    theta = np.deg2rad(degrees)
    rot = np.eye(4)
    rot[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return rot @ affine


AFFINES = {"exact LIA": LIA, "oblique LIA": tilted(LIA), "exact RAS": RAS, "oblique RAS": tilted(RAS)}


@pytest.mark.parametrize("name", AFFINES)
def test_affines_keep_their_axis_codes(name):
    assert "".join(nib.aff2axcodes(AFFINES[name])) == name[-3:]


@pytest.mark.parametrize(
    ("name", "orientation", "expected"),
    [
        ("exact LIA", "lia", True),
        ("exact LIA", "soft-lia", True),
        ("oblique LIA", "lia", False),
        ("oblique LIA", "soft-lia", True),
        ("exact LIA", "ras", False),
        ("exact LIA", "soft-ras", False),
        ("exact RAS", "ras", True),
        ("exact RAS", "soft-ras", True),
        ("oblique RAS", "ras", False),
        ("oblique RAS", "soft-ras", True),
        ("oblique LIA", "native", True),
        ("oblique LIA", None, True),
    ],
)
def test_is_conform_orientation(name, orientation, expected):
    img = nib.MGHImage(np.zeros(SHAPE, dtype=np.uint8), AFFINES[name])
    kwargs = {"vox_size": None, "img_size": None, "dtype": None, "verbose": False}
    assert is_conform(img, orientation=orientation, **kwargs) == expected


@pytest.mark.parametrize(
    ("name", "soft", "expected"),
    [("exact LIA", False, True), ("exact LIA", True, True), ("oblique LIA", False, False), ("oblique LIA", True, True)],
)
def test_is_orientation(name, soft, expected):
    assert is_orientation(AFFINES[name], "lia", soft=soft) == expected


@pytest.mark.parametrize("file_type", [nib.MGHImage, nib.Nifti1Image], ids=["mgz", "nii.gz"])
@pytest.mark.parametrize("source", ["exact LIA", "oblique LIA", "exact RAS"])
@pytest.mark.parametrize("orientation", ["lia", "soft-lia", "ras", "soft-ras"])
def test_conformed_image_is_conform(source, orientation, file_type, tmp_path):
    data = np.random.default_rng(0).uniform(0, 100, SHAPE).astype(np.float32)
    img = nib.MGHImage(data, AFFINES[source])
    kwargs = {"vox_size": 1.0, "img_size": 8, "dtype": np.uint8, "orientation": orientation}
    conformed = conform(img, file_type=file_type, **kwargs)
    assert is_conform(conformed, verbose=False, **kwargs)
    file = tmp_path / ("conformed.mgz" if file_type is nib.MGHImage else "conformed.nii.gz")
    nib.save(conformed, file)
    assert is_conform(nib.load(file), verbose=False, **kwargs)
