"""Regression checks for merged-label and robust intensity statistics of segstats.py and mri_segstats.py."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from FastSurferCNN.mri_segstats import make_arguments
from FastSurferCNN.segstats import calculate_merged_labels, pv_calc, robust_slice

with open(Path(__file__).parent / "data" / "segstats_robust.yaml") as fp:
    REFERENCE = yaml.safe_load(fp)


def make_images(runs: dict[int, list[list[int]]]) -> tuple[np.ndarray, np.ndarray]:
    """Segmentation and intensity image with the intensities `runs[label]` (see segstats_robust.yaml) per label."""
    values = {lab: np.concatenate([np.repeat(np.arange(r[0], r[1] + 1), r[2] if len(r) > 2 else 1) for r in label_runs])
              for lab, label_runs in runs.items()}
    seg = np.concatenate([np.full(len(v), lab, dtype=np.int32) for lab, v in values.items()])
    norm = np.concatenate(list(values.values())).astype(np.float32)
    pad = 16 * 16 * 8 - seg.size
    return np.pad(seg, (0, pad)).reshape(16, 16, 8), np.pad(norm, (0, pad)).reshape(16, 16, 8)


@pytest.mark.parametrize("groups", [([10, 12], [30, 32]), ([7, 7], [7]), ([11], []), ([], [])])
def test_merged_intensity_statistics_match_union(groups):
    arrays = {label: np.asarray(values, dtype=float) for label, values in enumerate(groups, 1)}
    counts = {label: values.size for label, values in arrays.items()}
    result = next(calculate_merged_labels(
        {99: [1, 2]}, counts, counts, counts,
        mins={label: values.min() for label, values in arrays.items() if values.size},
        maxes={label: values.max() for label, values in arrays.items() if values.size},
        sums={label: values.sum() for label, values in arrays.items()},
        sums_of_squares={label: np.square(values).sum() for label, values in arrays.items()},
    ))
    union = np.concatenate(list(arrays.values()))
    assert result["NVoxels"] == union.size
    assert result["Volume_mm3"] == union.size
    assert result["Mean"] == pytest.approx(union.mean() if union.size else 0)
    assert result["StdDev"] == pytest.approx(union.std(ddof=1) if union.size > 1 else 0)


@pytest.mark.parametrize("percent", list(REFERENCE["freesurfer"]))
def test_mri_segstats_robust_matches_freesurfer(percent):
    robust = [] if percent == "none" else ["--robust", percent]
    args = make_arguments().parse_args(["--seg", "seg.mgz", *robust])
    seg, norm = make_images(REFERENCE["intensities"])
    labels = list(REFERENCE["intensities"])

    table = pv_calc(seg, norm, norm, labels, robust_percentage=args.robust, merged_labels=REFERENCE["merged_labels"],
                    legacy_freesurfer=args.legacy_freesurfer)
    rows = {row["SegId"]: row for row in table}
    for label, expected in REFERENCE["freesurfer"][percent].items():
        actual = [rows[label][column] for column in REFERENCE["columns"]]
        assert actual == pytest.approx(expected, abs=1e-4), f"label {label}"
    # mri_segstats keeps no voxel of a single-voxel label and reports nan, keep the voxel instead
    assert (rows[6]["Mean"], rows[6]["StdDev"], rows[6]["Min"], rows[6]["Max"]) == (42, 0, 42, 42)


@pytest.mark.parametrize("percent", ["50", "-1"])
def test_mri_segstats_robust_rejects_invalid_percentages(percent):
    with pytest.raises(SystemExit):
        make_arguments().parse_args(["--seg", "seg.mgz", "--robust", percent])


@pytest.mark.parametrize(("nvoxels", "fraction", "expected"), [
    (20, 0.9, slice(1, 19)),         # (1 - 0.9) * 20 / 2 is 0.9999999999999998 in floating point
    (1000, 0.9, slice(50, 950)),
    (100, 0.96, slice(2, 98)),
    (7, 0.5, slice(1, 6)),
    (40, 1.0, slice(0, 40)),
    (3, 1e-9, slice(1, 2)),          # at least one voxel
])
def test_symmetric_robust_slice(nvoxels, fraction, expected):
    assert robust_slice(nvoxels, fraction) == expected
