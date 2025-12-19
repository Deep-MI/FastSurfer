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

import os
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
from scipy import ndimage

from CorpusCallosum.data import constants
from CorpusCallosum.shape.contour import CCContour
from CorpusCallosum.shape.postprocessing import recon_cc_surf_measure
from FastSurferCNN.utils import nibabelImage
from FastSurferCNN.utils.brainvolstats import mask_in_array

FSAVERAGE_PC_COORDINATE = np.array([131, 99])
FSAVERAGE_AC_COORDINATE = np.array([135, 130])


def smooth_contour(contour: tuple[np.ndarray, np.ndarray], window_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a contour using a moving average filter.

    Parameters
    ----------
    contour : tuple of arrays
        The contour coordinates (x, y).
    window_size : int
        Size of the smoothing window.

    Returns
    -------
    tuple of arrays
        The smoothed contour coordinates (x, y).
    
    """
    x, y = contour

    # Ensure the window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Create a padded version of the arrays to handle the edges
    x_padded = np.pad(x, (window_size//2, window_size//2), mode='wrap')
    y_padded = np.pad(y, (window_size//2, window_size//2), mode='wrap')

    # Apply moving average
    x_smoothed = np.zeros_like(x)
    y_smoothed = np.zeros_like(y)

    for i in range(len(x)):
        x_smoothed[i] = np.mean(x_padded[i:i+window_size])
        y_smoothed[i] = np.mean(y_padded[i:i+window_size])

    return (x_smoothed, y_smoothed)


def load_fsaverage_cc_template() -> CCContour:
    """Load and process the fsaverage corpus callosum template.

    This function loads the fsaverage segmentation from FreeSurfer's data directory,
    extracts the corpus callosum mask, and processes it to create a smooth template.

    Returns
    -------
    CCContour
        Object with all the contour information including:
        - contour : tuple[np.ndarray, np.ndarray] : x and y coordinates of the contour points.
        - anterior_endpoint_idx : np.ndarray : Index of the anterior endpoint.
        - posterior_endpoint_idx : np.ndarray : Index of the posterior endpoint.

    Raises
    ------
    OSError
        If FREESURFER_HOME environment variable is not set correctly.
    
    """
    # smooth outside contour
    # Apply smoothing to the outside contour using a moving average

    try:
        freesurfer_home = Path(os.environ['FREESURFER_HOME'])
    except KeyError as err:
        raise OSError(f"FREESURFER_HOME environment variable is not set correctly or does not exist: "
                      f"{freesurfer_home}, either provide your own template or set the "
                      f"FREESURFER_HOME environment variable") from err

    fsaverage_seg_path = freesurfer_home / 'subjects' / 'fsaverage' / 'mri' / 'aparc+aseg.mgz'
    fsaverage_seg = cast(nibabelImage, nib.load(fsaverage_seg_path))
    segmentation = np.asarray(fsaverage_seg.dataobj)

    midslice = segmentation.shape[0]//2 +1

    cc_mask = mask_in_array(segmentation[midslice], constants.SUBSEGMENT_LABELS)

    # Smooth the CC mask to reduce noise and irregularities

    # Apply binary closing to fill small holes
    cc_mask_smoothed = ndimage.binary_closing(cc_mask, structure=np.ones((3, 3)))

    # Apply binary opening to remove small isolated pixels
    cc_mask_smoothed = ndimage.binary_opening(cc_mask_smoothed, structure=np.ones((2, 2)))

    # Apply Gaussian smoothing and threshold to get a binary mask again
    cc_mask_smoothed = ndimage.gaussian_filter(cc_mask_smoothed.astype(float), sigma=0.8)
    cc_mask_smoothed = cc_mask_smoothed > 0.5

    # Use the smoothed mask for further processing
    cc_mask = cc_mask_smoothed.astype(int) * 192

    _, contour_with_thickness, (anterior_endpoint_idx, posterior_endpoint_idx) = recon_cc_surf_measure(
        segmentation=cc_mask[None],
        slice_idx=0,
        ac_coords_vox=FSAVERAGE_AC_COORDINATE,
        pc_coords_vox=FSAVERAGE_PC_COORDINATE,
        slice_lia_vox2midslice_ras=fsaverage_seg.affine,
        num_thickness_points=100,
        subdivisions=[1/6, 1/2, 2/3, 3/4],
        subdivision_method="shape",
        contour_smoothing=5,
        vox_size=(1., 1., 1.), # fsaverage is in 1mm isotropic
    )
    outside_contour = contour_with_thickness[:,:2].T

    # make sure the CC stays in shape despite smoothing by moving endpoints outwards
    outside_contour[0,anterior_endpoint_idx] -= 55
    outside_contour[0,posterior_endpoint_idx] += 30

    # Apply smoothing to the outside contour
    outside_contour_smoothed = smooth_contour(outside_contour, window_size=11)
    outside_contour_smoothed = smooth_contour(outside_contour_smoothed, window_size=15)
    outside_contour_smoothed = smooth_contour(outside_contour_smoothed, window_size=30)
    outside_contour = outside_contour_smoothed

    fsaverage_contour = CCContour(np.array(outside_contour).T, 
                                  np.zeros(len(outside_contour[0])), 
                                  endpoint_idxs=(anterior_endpoint_idx, posterior_endpoint_idx), 
                                  z_position=0.0)


    return fsaverage_contour
