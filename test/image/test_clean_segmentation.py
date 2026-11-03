"""Tests for the cleanup of the segmentation, which removes labels outside the brain mask and flips white matter
islands to their hemisphere before any output is derived from it (issue #956)."""

import numpy as np

from FastSurferCNN.reduce_to_aseg import clean_segmentation


def test_clean_segmentation():
    seg = np.zeros((64, 64, 64), dtype=np.int16)
    # left and right hemisphere: cortex parcels around white matter, touching in the middle
    seg[4:32, 8:40, 8:40] = 1003
    seg[20:30, 14:34, 14:34] = 2
    seg[32:60, 8:40, 8:40] = 2003
    seg[34:44, 14:34, 14:34] = 41
    # right white matter label in the left cortex, too far from any white matter to be decided by it
    island = np.zeros(seg.shape, dtype=bool)
    island[8:11, 22:25, 22:25] = True
    seg[island] = 41
    # white matter in the neck, too far from the brain for the mask to reach it
    neck = np.zeros(seg.shape, dtype=bool)
    neck[20:26, 54:60, 20:26] = True
    seg[neck] = 2

    cleaned, mask = clean_segmentation(seg, 1.0)

    assert not mask[neck].any() and not cleaned[neck].any()
    assert (cleaned[island] == 2).all()
    rest = ~(neck | island)
    np.testing.assert_array_equal(cleaned[rest], seg[rest])
