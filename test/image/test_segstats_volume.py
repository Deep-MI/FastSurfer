"""Checks that the partial volume corrected volumes of segstats.py do not depend on the intensity image (norm)."""

import numpy as np
import pytest

from FastSurferCNN.segstats import pv_calc

VOX_VOL = 0.8 ** 3


def make_ventricle_in_white_matter() -> tuple[np.ndarray, np.ndarray]:
    """Lateral ventricle (4) inside white matter (2), noisy partial volume guide, background in the outer two slices."""
    seg = np.full((24, 24, 24), 2, dtype=np.int32)
    seg[6:18, 9:15, 6:18] = 4
    seg[:2], seg[-2:] = 0, 0
    rng = np.random.default_rng(0)
    pv_guide = np.select([seg == 4, seg == 2], [30.0, 110.0], 0.0) + rng.normal(0, 1, seg.shape)
    return seg, pv_guide.astype(np.float32)


def volumes(table) -> dict[int, float]:
    """Volume per SegId of a pv_calc table."""
    return {row["SegId"]: row["Volume_mm3"] for row in table}


@pytest.mark.parametrize("legacy_freesurfer", [False, True])
@pytest.mark.parametrize(("labels", "merged_labels"), [
    ([2, 4], None),
    ([2, 4], {10000: [2, 4]}),
    # segstats.py --measure_only passes no labels, only the merged labels of the measures
    (np.zeros((0,), dtype=int), {10000: [4]}),
])
def test_volume_without_norm_matches_volume_with_norm(labels, merged_labels, legacy_freesurfer):
    seg, pv_guide = make_ventricle_in_white_matter()
    kwargs = {"vox_vol": VOX_VOL, "merged_labels": merged_labels, "legacy_freesurfer": legacy_freesurfer}
    without_norm = volumes(pv_calc(seg, pv_guide, None, labels, **kwargs))
    with_norm = volumes(pv_calc(seg, pv_guide, pv_guide, labels, **kwargs))
    assert without_norm == pytest.approx(with_norm, rel=1e-12)


def test_volume_without_norm_of_uniform_cube():
    seg = np.zeros((12, 12, 12), dtype=np.int32)
    seg[3:8, 3:8, 3:8] = 1
    pv_guide = (seg * 100).astype(np.float32)
    (row,) = pv_calc(seg, pv_guide, None, [1], vox_vol=VOX_VOL)
    assert row["NVoxels"] == 125
    assert row["Volume_mm3"] == pytest.approx(125 * VOX_VOL)
