from collections import OrderedDict
from functools import lru_cache
from logging import getLogger
from pathlib import Path

import nibabel.cmdline.diff
import numpy as np
import pytest
import yaml

from FastSurferCNN.utils.metrics import dice_score

from .common import SubjectDefinition, Tolerances

logger = getLogger(__name__)



@pytest.fixture(scope='module')
def segmentation_tolerances(segmentation_image: str) -> Tolerances:

    thresholds_file = Path(__file__).parent / f"data/thresholds/{segmentation_image}.yaml"
    assert thresholds_file.exists(), f"The thresholds file {thresholds_file} does not exist!"
    return Tolerances(thresholds_file)


@lru_cache
def read_image_type() -> dict:
    with open(Path(__file__).parent / "data/image.type.yaml") as fp:
        return yaml.safe_load(fp)


def compute_dice_score(test_data, reference_data, labels: dict[int, str]) -> tuple[float, dict[int, float]]:
    """
    Compute the dice score for each class (0 = no difference).

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
    dice_scores = {}
    logger.debug("Dice scores:")
    for _, (label, lname) in enumerate(labels.items()):
        dice_scores[label] = dice_score(np.equal(reference_data, label), np.equal(test_data, label), validate=False)
        logger.debug(f"Label {lname}: {dice_scores[label]:.4f}")
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

    # Load images
    test_file, test_img = test_subject.load_image(image)
    reference_file, reference_img = ref_subject.load_image(image)

    # Get the image headers
    headers = [test_img.header, reference_img.header]

    # Check the image headers
    header_diff = nibabel.cmdline.diff.get_headers_diff(headers)
    assert header_diff == OrderedDict(), f"Image headers do not match: {header_diff}"
    logger.debug("Image headers are same!")


def test_segmentation_image(
        test_subject: SubjectDefinition,
        ref_subject: SubjectDefinition,
        segmentation_image: str,
        segmentation_tolerances: Tolerances,
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

    Raises
    ------
    AssertionError
        If the dice score is not 0 for all classes
    """
    test_file, test_img = test_subject.load_image(segmentation_image)
    assert np.issubdtype(test_img.get_data_dtype(), np.integer), f"The image {segmentation_image} is not integer!"
    test_data = np.asarray(test_img.dataobj)
    reference_file, reference_img = ref_subject.load_image(segmentation_image)
    reference_data = np.asarray(reference_img.dataobj)

    label_segids = np.unique([reference_data, test_data])
    labels_lnames_tols = {lbl: segmentation_tolerances.threshold(lbl) for lbl in label_segids}
    labels_lnames = {k: v for k, (v, _) in labels_lnames_tols.items()}

    # Compute the dice score
    mean_dice, dice_scores = compute_dice_score(test_data, reference_data, labels_lnames)

    failed_labels = (i for i, dice in dice_scores.items() if not np.isclose(dice, 0, atol=labels_lnames_tols[i][1]))
    dice_exceeding_threshold = [f"Label {labels_lnames[lbl]}: {1-dice_scores[lbl]}" for lbl in failed_labels]
    assert [] == dice_exceeding_threshold, f"Dice scores in {segmentation_image} are not within range!"
    logger.debug("Dice scores are within range for all classes")


def test_intensity_image(test_subject: SubjectDefinition, ref_subject: SubjectDefinition, intensity_image: str):
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

    Raises
    ------
    AssertionError
        If the mean square error is not 0
    """
    # Get the image data
    test_file, test_img = test_subject.load_image(intensity_image)
    test_data = test_img.get_fdata()
    reference_file, reference_img = ref_subject.load_image(intensity_image)
    reference_data = reference_img.get_fdata()
    # Check the image data
    np.testing.assert_allclose(test_data, reference_data, rtol=1e-4, err_msg="Image intensity data do not match!")

    logger.debug("Image data matches!")


def pytest_generate_tests(metafunc):
    images_by_type = {}
    if any(f in metafunc.fixturenames for f in ("intensity_image", "segmentation_image", "image")):
        images_by_type = read_image_type()

    for typ in ["intensity", "segmentation"]:
        if f"{typ}_image" in metafunc.fixturenames:
            metafunc.parametrize(f"{typ}_image", images_by_type[typ], scope="module")
    if "image" in metafunc.fixturenames:
        from itertools import chain
        all_images = chain(images_by_type["intensity"], images_by_type["segmentation"])
        metafunc.parametrize("image", all_images, scope="module")
