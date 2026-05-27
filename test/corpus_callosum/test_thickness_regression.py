import numpy as np

from CorpusCallosum.shape.contour import CCContour
from CorpusCallosum.shape.thickness import insert_point_with_thickness


def test_insert_point_with_thickness_reuses_existing_point() -> None:
    contour = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
    )
    thickness = np.array([np.nan, 1.0, 2.0, 3.0])

    new_contour, new_thickness, point_idx, inserted = insert_point_with_thickness(
        contour,
        thickness,
        np.array([0.0, 0.0]),
        4.0,
        return_index=True,
    )

    assert inserted is False
    assert point_idx == 0
    np.testing.assert_array_equal(new_contour, contour)
    assert new_thickness[0] == 4.0


def test_fill_thickness_values_uses_zero_distance_match() -> None:
    contour = CCContour(
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ),
        np.array([1.0, 2.0, np.nan, 4.0]),
    )

    contour.fill_thickness_values()

    assert contour.thickness_values[2] == 2.0
    assert np.all(np.isfinite(contour.thickness_values))
