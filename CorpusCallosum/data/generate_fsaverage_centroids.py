#!/usr/bin/env python3
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
"""
Script to generate static fsaverage centroids file.

This script extracts centroids from the fsaverage template segmentation
and saves them to a JSON file for fast loading during pipeline execution.
Run this script once to generate the centroids file.
"""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from read_write import convert_numpy_to_json_serializable, get_centroids_from_nib

import FastSurferCNN.utils.logging as logging

logger = logging.get_logger(__name__)


def main() -> None:
    """Generate and save fsaverage centroids to a static file.

    This script extracts centroids from the fsaverage template segmentation
    and saves them to a JSON file for fast loading during pipeline execution.

    The script performs the following steps:
    1. Load fsaverage segmentation from FreeSurfer directory
    2. Extract centroids for all anatomical structures
    3. Save centroids to JSON file
    4. Extract and save affine matrix and header fields

    Raises
    ------
    OSError
        If FREESURFER_HOME environment variable is not set or invalid
    FileNotFoundError
        If required fsaverage files are not found

    Notes
    -----
    The script saves two files:
    - fsaverage_centroids.json : Contains centroids for each anatomical structure
    - fsaverage_data.json : Contains affine matrix and header information
    """
    
    # Get fsaverage path from FreeSurfer environment
    try:
        fs_home = Path(os.environ['FREESURFER_HOME'])
        if not fs_home.exists():
            raise OSError(f"FREESURFER_HOME environment variable is not set correctly or does not exist: {fs_home}")
        
        fsaverage_path = fs_home / 'subjects' / 'fsaverage'
        if not fsaverage_path.exists():
            raise OSError(f"fsaverage path does not exist: {fsaverage_path}")
        
        fsaverage_aseg_path = fsaverage_path / 'mri' / 'aseg.mgz'
        if not fsaverage_aseg_path.exists():
            raise FileNotFoundError(f"fsaverage aseg file does not exist: {fsaverage_aseg_path}")
            
    except KeyError as err:
        raise OSError("FREESURFER_HOME environment variable is not set") from err
    
    print(f"Loading fsaverage segmentation from: {fsaverage_aseg_path}")
    
    # Load fsaverage segmentation
    fsaverage_nib = nib.load(fsaverage_aseg_path)
    
    # Extract centroids
    print("Extracting centroids from fsaverage...")
    centroids_dst = get_centroids_from_nib(fsaverage_nib)
    
    print(f"Found {len(centroids_dst)} anatomical structures with centroids")
    
    # Convert to JSON-serializable format
    centroids_serializable = convert_numpy_to_json_serializable(centroids_dst)
    
    # Save centroids to JSON file
    centroids_output_path = Path(__file__).parent / "fsaverage_centroids.json"
    logger.info(f"Saving fsaverage centroids to {centroids_output_path}")
    with open(centroids_output_path, 'w') as f:
        json.dump(centroids_serializable, f, indent=2)
    
    print(f"Fsaverage centroids saved to: {centroids_output_path}")
    print(f"Centroids file size: {centroids_output_path.stat().st_size} bytes")
    
    # Extract and save fsaverage affine matrix and header fields
    print("Extracting fsaverage affine matrix and header fields...")
    fsaverage_affine = fsaverage_nib.affine.astype(float)  # Convert to float for JSON serialization
    
    # Extract header fields needed for LTA
    header = fsaverage_nib.header
    dims = [int(x) for x in header.get_data_shape()[:3]]  # Convert to int for JSON serialization
    delta = [float(x) for x in header.get_zooms()[:3]]  # Convert to float for JSON serialization
    vox2ras = header.get_vox2ras()
    
    # Direction cosines matrix (Mdc) - extract rotation part without scaling
    delta_diag = np.diag(delta)
    # Avoid division by zero by using a small epsilon for zero values
    delta_safe = np.where(delta_diag == 0, 1e-10, delta_diag)
    Mdc = (vox2ras[:3, :3] / delta_safe).astype(float)  # Convert to float for JSON serialization
    
    Pxyz_c = vox2ras[:3, 3].astype(float)  # Convert to float for JSON serialization
    
    # Combine affine and header data
    combined_data = {
        "affine": fsaverage_affine.tolist(),  # Convert numpy array to list for JSON serialization
        "vox2ras_tkr": fsaverage_nib.header.get_vox2ras_tkr().tolist(),
        "header": {
            "dims": dims,
            "delta": delta,
            "Mdc": Mdc.tolist(),  # Convert numpy array to list for JSON serialization
            "Pxyz_c": Pxyz_c.tolist()  # Convert numpy array to list for JSON serialization
        }
    }
    
    # Convert the entire structure to JSON-serializable format to handle any remaining numpy types
    combined_data_serializable = convert_numpy_to_json_serializable(combined_data)
    
    # Save combined data to JSON file
    combined_output_path = Path(__file__).parent / "fsaverage_data.json"
    logger.info(f"Saving fsaverage affine and header data to {combined_output_path}")
    with open(combined_output_path, 'w') as f:
        json.dump(combined_data_serializable, f, indent=2)
    
    print(f"Fsaverage affine and header data saved to: {combined_output_path}")
    print(f"Combined file size: {combined_output_path.stat().st_size} bytes")
    print(f"Affine matrix shape: {fsaverage_affine.shape}")
    print(f"Header dims: {dims}, delta: {delta}")
    
    # Print some statistics
    label_ids = list(centroids_dst.keys())
    print(f"Label IDs range: {min(label_ids)} to {max(label_ids)}")
    print("Sample centroids:")
    for label_id in sorted(label_ids)[:5]:
        centroid = centroids_dst[label_id]
        print(f"  Label {label_id}: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
    
    print("Fsaverage affine matrix:")
    print(fsaverage_affine)
    
    print("Fsaverage header fields:")
    print(f"  dims: {dims}")
    print(f"  delta: {delta}")
    print(f"  Mdc shape: {Mdc.shape}")
    print(f"  Pxyz_c: {Pxyz_c}")
    print("Combined data structure created successfully")


if __name__ == "__main__":
    main()
