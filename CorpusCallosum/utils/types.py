from typing import Literal, TypedDict

from numpy import dtype, ndarray

from FastSurferCNN.utils import ScalarType

__all__ = [
    "CCMeasuresDict",
    "ContourList",
    "ContourThickness",
    "Points2dType",
    "Points3dType",
    "Polygon2dType",
    "Polygon3dType",
    "SliceSelection",
    "SubdivisionMethod",
]

Polygon2dType = ndarray[tuple[Literal[2], int], dtype[ScalarType]]
Polygon3dType = ndarray[tuple[Literal[3], int], dtype[ScalarType]]
Points2dType = ndarray[tuple[int, Literal[2]], dtype[ScalarType]]
Points3dType = ndarray[tuple[int, Literal[3]], dtype[ScalarType]]
ContourList = list[Polygon2dType]
ContourThickness = ndarray[tuple[Literal[3], int], dtype[ScalarType]]
SliceSelection = Literal["middle", "all"] | int
SubdivisionMethod = Literal["shape", "vertical", "angular", "eigenvector"]

class CCMeasuresDict(TypedDict):
    """TypedDict for corpus callosum measures.

    Attributes
    ----------
    cc_index : float
        Corpus callosum shape index.
    circularity : float
        Shape circularity measure.
    areas : np.ndarray
        Areas of subdivided regions.
    midline_length : float
        Length along the midline.
    thickness : float
        Array of thickness measurements.
    curvature : float
        Array of curvature measurements.
    thickness_profile : np.ndarray of type float
        Thickness measurements along the contour.
    total_area : float
        Total area of the CC.
    total_perimeter : float
        Total perimeter length.
    split_contours : list of np.ndarray
        Subdivided contour segments in AS-slice coordinates.
    midline_equidistant : np.ndarray
        Equidistant points along midline in AS-slice coordinates.
    levelpaths : list of np.ndarray
        Paths for thickness measurements in AS-slice coordinates.
    slice_index : int
        Index of the processed slice.
    """
    cc_index: float
    circularity: float
    areas: ndarray
    midline_length: float
    thickness: float
    curvature: float
    thickness_profile: ndarray[tuple[int], dtype[float]]
    total_area: float
    total_perimeter: float
    total_area: float
    total_perimeter: float
    split_contours: ContourList
    midline_equidistant: ndarray
    levelpaths: list[ndarray]
    slice_index: int
