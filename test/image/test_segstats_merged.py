"""Regression checks for merged-region and trimmed intensity statistics."""

import numpy as np
import pytest

from FastSurferCNN.segstats import calculate_merged_labels, global_stats


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


@pytest.mark.parametrize("fraction", [0.5, 0.95, 1.0])
def test_trimmed_merged_statistics_use_retained_intensities(fraction):
    segmentation = np.repeat([1, 2], 4).reshape(2, 2, 2)
    norm = np.asarray([0, 10, 12, 100, 5, 30, 32, 200]).reshape(2, 2, 2)
    values = dict(global_stats(label, norm, segmentation, robust_percentage=fraction) for label in (1, 2))
    result = next(calculate_merged_labels(
        {99: [1, 2]}, {label: stats[0] for label, stats in values.items()},
        {label: stats[1] for label, stats in values.items()}, {1: 4, 2: 4},
        mins={label: stats[2] for label, stats in values.items()},
        maxes={label: stats[3] for label, stats in values.items()},
        sums={label: stats[4] for label, stats in values.items()},
        sums_of_squares={label: stats[5] for label, stats in values.items()},
    ))
    trim = int((1 - fraction) * 4 / 2)
    retained = np.concatenate([np.sort(norm[segmentation == label])[trim:4 - trim] for label in (1, 2)])
    assert result["NVoxels"] == 8
    assert result["Volume_mm3"] == 8
    assert result["Mean"] == pytest.approx(retained.mean())
    assert result["StdDev"] == pytest.approx(retained.std(ddof=1))
    assert result["Min"] == retained.min()
    assert result["Max"] == retained.max()
