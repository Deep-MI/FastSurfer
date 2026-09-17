from functools import lru_cache
from logging import getLogger
from pathlib import Path

import numpy as np
import pytest
import yaml

from FastSurferCNN.utils.metrics import dice_score

from .common import SubjectDefinition, Tolerances, chain_order, skip_if_missing, write_table_file
from .helper import Approx, assert_same_headers

logger = getLogger(__name__)


@pytest.fixture(scope='module')
def segmentation_tolerances(segmentation_image: str) -> Tolerances:

    thresholds_file = Path(__file__).parent / f"data/{segmentation_image}.yaml"
    assert thresholds_file.exists(), f"The thresholds file {thresholds_file} does not exist!"
    return Tolerances(thresholds_file)


@lru_cache
def read_image_intensity_thresholds() -> dict:
    with open(Path(__file__).parent / "data/image.intensity.yaml") as fp:
        return yaml.safe_load(fp)["thresholds"]


def compute_dice_score(test_data, reference_data, labels: dict[int, str]) -> tuple[float, dict[int, float]]:
    """
    Compute the dice score for each class (1 = no difference).

    Parameters
    ----------
    test_data : np.ndarray
        Test image data.
    reference_data : np.ndarray
        Reference image data.
    labels : np.ndarray
        Unique labels in the image data.

    Returns
    -------
    float
        The average dice score for all classes.
    dict[int, float]
        Dice scores for each class.
    """

    def __calc_dice(label: int) -> float:
        return dice_score(np.equal(reference_data, label), np.equal(test_data, label), validate=False)

    label_iter = labels.keys() if isinstance(labels, dict) else labels
    dice_scores = {label: __calc_dice(label) for label in label_iter}
    mean_dice_score = np.asarray(list(dice_scores.values())).mean()
    return mean_dice_score, dice_scores


def test_image_headers(test_subject: SubjectDefinition, ref_subject: SubjectDefinition, image: str):
    """
    Test the image headers by comparing the headers of the test and reference images.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    ref_subject : SubjectDefinition
        Definition of the reference subject.
    image: str
        Name image file to check the headers of.

    Raises
    ------
    AssertionError
        If the image headers do not match
    """

    skip_if_missing(ref_subject, test_subject, image)

    # Load images
    test_file, test_img = test_subject.load_image(image)
    reference_file, reference_img = ref_subject.load_image(image)

    assert_same_headers(test_img.header, reference_img.header)


def test_segmentation_image(
        test_subject: SubjectDefinition,
        ref_subject: SubjectDefinition,
        segmentation_image: str,
        segmentation_tolerances: Tolerances,
        pytestconfig: pytest.Config,
):
    """
    Test the segmentation data by calculating and comparing dice scores.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    ref_subject : SubjectDefinition
        Definition of the reference subject.
    segmentation_image : str
        Name of the segmentation image file.
    segmentation_tolerances: Tolerances
        Object to provide the relevant tolerances for the respective segmentation_image.
    pytestconfig : pytest.Config
        The sessions config object.

    Raises
    ------
    AssertionError
        If the dice score is not 1 for all classes
    """
    skip_if_missing(ref_subject, test_subject, segmentation_image)

    test_file, test_img = test_subject.load_image(segmentation_image)
    assert np.issubdtype(test_img.get_data_dtype(), np.integer), f"The image {segmentation_image} is not integer!"
    test_data = np.asarray(test_img.dataobj)
    reference_file, reference_img = ref_subject.load_image(segmentation_image)
    reference_data = np.asarray(reference_img.dataobj)

    label_segids = np.unique([reference_data, test_data]).tolist()
    labels_lnames_tols = {lbl: segmentation_tolerances.threshold(lbl) for lbl in label_segids}
    labels_lnames = {k: v for k, (v, _) in labels_lnames_tols.items()}

    def is_low_dice(label: int, score: float) -> bool:
        # the tolerance is the distance from a perfect overlap that is still accepted
        return not np.isclose(score, 1, atol=labels_lnames_tols[label][1])

    # Compute the dice score
    _, dice_scores = compute_dice_score(test_data, reference_data, labels_lnames)

    delta_dir: Path = pytestconfig.getoption("--collect_csv")
    if delta_dir:
        delta_dir.mkdir(parents=True, exist_ok=True)
        write_table_file(delta_dir / "dice.csv", test_subject.name, segmentation_image, dice_scores)

    # report the limit as the lowest overlap that still passes, so it can be read against the value
    failed_labels = [(i, labels_lnames_tols[i]) for i, dice in dice_scores.items() if is_low_dice(i, dice)]
    dice_exceeding_threshold = [
        f"{lname}: Dice {dice_scores[lbl]:.4f} (min {1 - tol:.4f})" for lbl, (lname, tol) in failed_labels
    ]
    summary = ""
    if failed_labels:
        worst_lbl, (worst_lname, worst_tol) = min(failed_labels, key=lambda item: dice_scores[item[0]])
        summary = (
            f" {len(failed_labels)} of {len(dice_scores)} labels below their minimum, worst {worst_lname} at "
            f"Dice {dice_scores[worst_lbl]:.4f} (min {1 - worst_tol:.4f})."
        )
    assert dice_exceeding_threshold == [], f"Dice scores in {segmentation_image} are not within range!{summary}"
    logger.debug("Dice scores are within range for all classes")


def test_intensity_image(
        test_subject: SubjectDefinition,
        ref_subject: SubjectDefinition,
        intensity_image: str,
        pytestconfig: pytest.Config,
):
    """
    Test the intensity data by calculating and comparing the mean square error.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    ref_subject : SubjectDefinition
        Definition of the reference subject.
    intensity_image : str
        Name of the image file.
    pytestconfig : pytest.Config
        The sessions config object.

    Raises
    ------
    AssertionError
        If the mean square error is not 0
    """
    skip_if_missing(ref_subject, test_subject, intensity_image)

    # Get the image data
    test_file, test_img = test_subject.load_image(intensity_image)
    test_data = test_img.get_fdata()
    reference_file, reference_img = ref_subject.load_image(intensity_image)
    reference_data = reference_img.get_fdata()

    delta_dir = pytestconfig.getoption("--collect_csv")
    if delta_dir:
        delta_dir.mkdir(parents=True, exist_ok=True)
        # this analysis will write not only the max (tested below, but also mean, mse and some percentiles)
        absdelta = np.abs(test_data - reference_data)
        scores = {p: v for p, v in zip(("median", "95th", "99th"), np.percentile(absdelta, (50, 95, 99)), strict=True)}
        scores.update(mean=absdelta.mean(), max=np.max(absdelta))
        reldelta = absdelta / np.maximum(np.maximum(abs(reference_data), abs(test_data)), 1e-8)
        scores.update(rel=reldelta.mean(), relmax=np.max(reldelta))
        write_table_file(delta_dir / "intensity.csv", test_subject.name, intensity_image, scores)

    rtol = read_image_intensity_thresholds()[intensity_image]
    # Check the image data
    assert test_data == Approx(reference_data, rel=rtol), "Image intensity data do not match!"

    logger.debug("Image data matches!")


def pytest_generate_tests(metafunc: pytest.Metafunc):
    # every list is ordered by the pipeline, so the first failure reported is the first divergence
    # and the ones after it are its consequences rather than separate findings
    intensity_files = []
    if any(f in metafunc.fixturenames for f in ("intensity_image", "image")):
        intensity_thresholds = read_image_intensity_thresholds()
        intensity_files = sorted(intensity_thresholds.keys(), key=chain_order)
        if "intensity_image" in metafunc.fixturenames:
            metafunc.parametrize("intensity_image", intensity_files, scope="module")

    segmentation_files = []
    if any(f in metafunc.fixturenames for f in ("segmentation_image", "image")):
        __files = (Path(__file__).parent / "data").glob("*.yaml")
        segmentation_files = sorted(
            (f.stem for f in __files if f.stem.endswith((".nii", ".nii.gz", ".mgz"))),
            key=chain_order,
        )
        if "segmentation_image" in metafunc.fixturenames:
            metafunc.parametrize("segmentation_image", segmentation_files, scope="module")

    if "image" in metafunc.fixturenames:
        all_images = sorted(set(intensity_files) | set(segmentation_files), key=chain_order)
        metafunc.parametrize("image", all_images, scope="module")
