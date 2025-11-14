# Copyright 2025 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import TypedDict

import nibabel as nib
import numpy as np
from numpy import typing as npt

import FastSurferCNN.utils.logging as logging


class FSAverageHeader(TypedDict):
    dims: npt.NDArray[int]
    delta: npt.NDArray[float]
    Mdc: npt.NDArray[float]
    Pxyz_c: npt.NDArray[float]

logger = logging.get_logger(__name__)


def get_centroids_from_nib(seg_img: nib.analyze.SpatialImage, label_ids: list[int] | None = None) \
        -> dict[int, np.ndarray | None]:
    """Get centroids of segmentation labels in RAS coordinates.

    Parameters
    ----------
    seg_img : nibabel.analyze.SpatialImage
        Input segmentation image.
    label_ids : list[int], optional
        List of label IDs to extract centroids for. If None, extracts all non-zero labels.

    Returns
    -------
    dict[int, np.ndarray | None]
        A dict mapping label IDs to their centroids (x,y,z) in RAS coordinates, None if label did not exist.
    """
    # Get segmentation data and affine
    seg_data: npt.NDArray[np.integer] = np.asarray(seg_img.dataobj)
    vox2ras: npt.NDArray[float] = seg_img.affine
    
    # Get unique labels
    if label_ids is None:
        labels = np.unique(seg_data)
        labels = labels[labels > 0]  # Exclude background
    else:
        labels = label_ids
    
    def _calc_ras_centroid(mask_vox: npt.NDArray[np.integer]) -> npt.NDArray[float]:
        # Calculate centroid in voxel space
        vox_centroid = np.mean(mask_vox, axis=1, dtype=float)

        # Convert to homogeneous coordinates
        vox_centroid = np.append(vox_centroid, 1)

        # Transform to RAS coordinates and return without homogeneous coordinate
        return (vox2ras @ vox_centroid)[:3]

    centroids = {}
    for label in labels:
        # Get voxel indices for this label
        vox_coords = np.array(np.where(seg_data == label))
        centroids[int(label)] = None if vox_coords.size == 0 else _calc_ras_centroid(vox_coords)
        
    return centroids


def convert_numpy_to_json_serializable(obj: object) -> object:
    """Convert numpy types to JSON serializable types.

    Parameters
    ----------
    obj : object
        Object to convert to JSON serializable type.

    Returns
    -------
    object
        JSON serializable version of the input object.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        # Handle numpy scalar types
        return obj.item()
    else:
        return obj


def load_fsaverage_centroids(centroids_path: str | Path) -> dict[int, npt.NDArray[float]]:
    """Load fsaverage centroids from static JSON file.

    Parameters
    ----------
    centroids_path : str or Path
        Path to the JSON file containing centroids.

    Returns
    -------
    dict[int, np.ndarray]
        Dictionary mapping label IDs to their centroids in RAS coordinates.
    """
    
    centroids_path = Path(centroids_path)
    if not centroids_path.exists():
        raise FileNotFoundError(f"Fsaverage centroids file not found: {centroids_path}")
    
    with open(centroids_path) as f:
        centroids_data = json.load(f)
    
    # Convert string keys back to integers and lists back to numpy arrays
    return {int(label): np.array(centroid) for label, centroid in centroids_data.items()}


def load_fsaverage_affine(affine_path: str | Path) -> npt.NDArray[float]:
    """Load fsaverage affine matrix from static text file.

    Parameters
    ----------
    affine_path : str or Path
        Path to the text file containing affine matrix.

    Returns
    -------
    np.ndarray
        4x4 affine transformation matrix.
    """
    
    affine_path = Path(affine_path)
    if not affine_path.exists():
        raise FileNotFoundError(f"Fsaverage affine file not found: {affine_path}")
    
    affine_matrix = np.loadtxt(affine_path).astype(float)
    
    if affine_matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine matrix, got shape {affine_matrix.shape}")
    
    return affine_matrix


def load_fsaverage_data(data_path: str | Path) -> tuple[npt.NDArray[float], FSAverageHeader, npt.NDArray[float]]:
    """Load fsaverage affine matrix and header fields from static JSON file.

    Parameters
    ----------
    data_path : str or Path
        Path to the JSON file containing combined data.

    Returns
    -------
    affine_matrix : np.ndarray
        4x4 affine transformation matrix.
    header_fields : dict
        Header fields needed for LTA:
            - dims : list[int]
                Volume dimensions [x,y,z].
            - delta : list[float]
                Voxel size in mm [x,y,z].
            - Mdc : np.ndarray
                3x3 direction cosines matrix.
            - Pxyz_c : np.ndarray
                RAS center coordinates [x,y,z].
    vox2ras_tkr : np.ndarray
        Voxel to RAS tkr-space transformation matrix.

    Raises
    ------
    FileNotFoundError
        If the data file doesn't exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    ValueError
        If required fields are missing.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Fsaverage data file not found: {data_path}")
    
    with open(data_path) as f:
        data = json.load(f)
    
    # Verify required fields
    if "affine" not in data:
        raise ValueError("Required field 'affine' missing from data file")
    if "header" not in data:
        raise ValueError("Required field 'header' missing from data file")
    
    required_header_fields = ["dims", "delta", "Mdc", "Pxyz_c"]
    for field in required_header_fields:
        if field not in data["header"]:
            raise ValueError(f"Required header field missing: {field}")
    
    # Convert lists back to numpy arrays
    affine_matrix = np.array(data["affine"])
    vox2ras_tkr = np.array(data["vox2ras_tkr"])
    header_data = FSAverageHeader(
        dims=data["header"]["dims"],
        delta=data["header"]["delta"],
        Mdc=np.array(data["header"]["Mdc"]),
        Pxyz_c=np.array(data["header"]["Pxyz_c"]),
    )
    
    # Validate affine matrix shape
    if affine_matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine matrix, got shape {affine_matrix.shape}")
    
    return affine_matrix, header_data, vox2ras_tkr