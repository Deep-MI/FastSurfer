import os
import pytest
from pathlib import Path
import nibabel as nib
import nibabel.cmdline.diff
import numpy as np
import yaml

from collections import OrderedDict

from .common import *

from CerebNet.utils.metrics import dice_score

from logging import getLogger

logger = getLogger(__name__)


def load_image(subject_path, image_name):
    """
    Load the image data using nibabel.

    Parameters
    ----------
    subject_path : str
        Path to the subject directory.
    image_name : str
        Name of the image file.

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        Image data.
    """
    image_path = os.path.join(subject_path, "mri", image_name)
    image = nib.load(image_path)

    return image


def compute_dice_score(test_data, reference_data, labels):
    """
    Compute the dice score for each class.

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
    np.ndarray
        Dice scores for each class.
    """

    # Classes
    num_classes = len(labels)

    dscore = np.zeros(shape=num_classes)

    for idx in range(num_classes):
        current_label = labels[idx]

        pred = (test_data == current_label).astype(int)
        gt = (reference_data == current_label).astype(int)

        dscore[idx] = dice_score(pred, gt)

    logger.debug("\nDice score: ", dscore)

    return dscore


def compute_mean_square_error(test_data, reference_data):
    """
    Compute the mean square error between the test and reference data.

    Parameters
    ----------
    test_data : np.ndarray
        Test image data.
    reference_data : np.ndarray
        Reference image data.

    Returns
    -------
    float
        Mean square error.
    """

    mse = ((test_data - reference_data) ** 2).mean()
    logger.debug("\nMean square error: ", mse)

    return mse


@pytest.mark.parametrize("test_subject", load_test_subjects())
def test_image_headers(subjects_dir, test_dir, reference_dir, test_subject):
    """
    Test the image headers by comparing the headers of the test and reference images.

    Parameters
    ----------
    subjects_dir : str
        Path to the subjects directory.
    test_dir : str
        Name of test directory.
    reference_dir: str
        Name of reference directory.
    test_subject : str
        Name of the test subject.

    Raises
    ------
    AssertionError
        If the image headers do not match
    """

    # Load images
    test_subject = os.path.join(subjects_dir, test_dir, test_subject)
    test_image = load_image(test_subject, "brain.mgz")
    reference_subject = os.path.join(subjects_dir, reference_dir, test_subject)
    reference_image = load_image(reference_subject, "brain.mgz")

    # Get the image headers
    headers = [test_image.header, reference_image.header]

    # Check the image headers
    header_diff = nibabel.cmdline.diff.get_headers_diff(headers)
    assert header_diff == OrderedDict(), f"Image headers do not match: {header_diff}"
    logger.debug("Image headers are correct")


@pytest.mark.parametrize("test_subject", load_test_subjects())
def test_seg_data(subjects_dir, test_dir, reference_dir, test_subject):
    """
    Test the segmentation data by calculating and comparing dice scores.

    Parameters
    ----------
    subjects_dir : str
        Path to the subjects directory.
    test_dir : str
        Name of test directory.
    reference_dir : str
        Name of reference directory.
    test_subject : str
        Name of the test subject.

    Raises
    ------
    AssertionError
        If the dice score is not 0 for all classes
    """

    test_file = os.path.join(subjects_dir, test_dir, test_subject)
    test_image = load_image(test_file, "aseg.mgz")

    reference_subject = os.path.join(subjects_dir, reference_dir, test_subject)
    reference_image = load_image(reference_subject, "aseg.mgz")

    labels = np.unique([np.asarray(reference_image.dataobj), np.asarray(test_image.dataobj)])

    # Get the image data
    test_data = np.asarray(test_image.dataobj)
    reference_data = np.asarray(reference_image.dataobj)

    # Compute the dice score
    dscore = compute_dice_score(test_data, reference_data, labels)

    # Check the dice score
    np.testing.assert_allclose(dscore, 0, atol=1e-6, rtol=1e-6,
                               err_msg=f"Dice scores are not within range for all classes")

    # assert dscore == 1, "Dice scores are not 1 for all classes"

    logger.debug("Dice scores are within range for all classes")


@pytest.mark.parametrize("test_subject", load_test_subjects())
def test_int_data(subjects_dir, test_dir, reference_dir, test_subject):
    """
    Test the intensity data by calculating and comparing the mean square error.

    Parameters
    ----------
    subjects_dir : str
        Path to the subjects directory.
    test_dir : str
        Name of test directory.
    reference_dir : str
        Name of reference directory.
    test_subject : str
        Name of the test subject.

    Raises
    ------
    AssertionError
        If the mean square error is not 0
    """

    test_file = os.path.join(subjects_dir, test_dir, test_subject)
    test_image = load_image(test_file, "brain.mgz")

    reference_subject = os.path.join(subjects_dir, reference_dir, test_subject)
    reference_image = load_image(reference_subject, "brain.mgz")

    # Get the image data
    test_data = test_image.get_fdata()
    reference_data = reference_image.get_fdata()

    mse = compute_mean_square_error(test_data, reference_data)

    # Check the image data
    assert mse == 0, "Mean square error is not 0"

    logger.debug("\nImage data matches")
