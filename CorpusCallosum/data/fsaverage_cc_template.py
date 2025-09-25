import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from shape.cc_postprocessing import process_slice


def smooth_contour(contour, window_size=5):
        """
        Smooth a contour using a moving average filter
        
        Parameters:
        -----------
        contour : tuple of arrays
            The contour coordinates (x, y)
        window_size : int
            Size of the smoothing window
            
        Returns:
        --------
        tuple of arrays
            The smoothed contour coordinates (x, y)
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

def load_fsaverage_cc_template():
    # smooth outside contour
    # Apply smoothing to the outside contour using a moving average

    try:
        freesurfer_home = Path(os.environ['FREESURFER_HOME'])
    except KeyError as err:
        raise OSError(f"FREESURFER_HOME environment variable is not set correctly or does not exist: "
                      f"{freesurfer_home}, either provide your own template or set the "
                      f"FREESURFER_HOME environment variable") from err

    fsaverage_seg_path = freesurfer_home / 'subjects' / 'fsaverage' / 'mri' / 'aparc+aseg.mgz'
    fsaverage_seg = nib.load(fsaverage_seg_path)
    segmentation = fsaverage_seg.get_fdata()

    PC = np.array([131, 99])
    AC = np.array([135, 130])


    midslice = segmentation.shape[0]//2 +1

    cc_mask = segmentation[midslice] == 251
    cc_mask |= segmentation[midslice] == 252
    cc_mask |= segmentation[midslice] == 253
    cc_mask |= segmentation[midslice] == 254
    cc_mask |= segmentation[midslice] == 255

    # Smooth the CC mask to reduce noise and irregularities

    # Apply binary closing to fill small holes
    cc_mask_smoothed = ndimage.binary_closing(cc_mask, structure=np.ones((3, 3)))

    # Apply binary opening to remove small isolated pixels
    cc_mask_smoothed = ndimage.binary_opening(cc_mask_smoothed, structure=np.ones((2, 2)))

    # Apply Gaussian smoothing and threshold to get a binary mask again
    cc_mask_smoothed = ndimage.gaussian_filter(cc_mask_smoothed.astype(float), sigma=0.8)
    cc_mask_smoothed = cc_mask_smoothed > 0.5

    # Use the smoothed mask for further processing
    cc_mask = cc_mask_smoothed.astype(int)
    cc_mask[cc_mask > 0] = 192

    (_, contour_with_thickness, anterior_endpoint_idx, 
     posterior_endpoint_idx) = process_slice(segmentation=cc_mask[None], 
                                             slice_idx=0, 
                                             ac_coords=AC, 
                                             pc_coords=PC, 
                                             affine=fsaverage_seg.affine, 
                                             num_thickness_points=100, 
                                             subdivisions=[1/6, 1/2, 2/3, 3/4], 
                                             subdivision_method="shape", 
                                             contour_smoothing=1.0)
    outside_contour = contour_with_thickness[0].T


    outside_contour[0][anterior_endpoint_idx] -= 55
    outside_contour[0][posterior_endpoint_idx] += 30

    # Apply smoothing to the outside contour
    outside_contour_smoothed = smooth_contour(outside_contour, window_size=11)
    outside_contour_smoothed = smooth_contour(outside_contour_smoothed, window_size=15)
    outside_contour_smoothed = smooth_contour(outside_contour_smoothed, window_size=30)
    outside_contour = outside_contour_smoothed


    # Plot CC contour with levelsets

    # midline_equidistant = output_dict['midline_equidistant']
    # levelpaths = output_dict['levelpaths']
    # plt.figure(figsize=(12, 8))

    # plt.plot(outside_contour[0], outside_contour[1], 'k-', linewidth=2)

    # # Plot the midline
    # if midline_equidistant is not None:
    #     midline_x, midline_y = zip(*midline_equidistant)
    #     plt.plot(midline_x, midline_y, 'r-', linewidth=2, label='Midline')

    # # Plot the level paths
    # if levelpaths:
    #     for i, path in enumerate(levelpaths):
    #         path_x, path_y = path[:,0], path[:,1]
    #         plt.plot(path_x, path_y, 'g--', linewidth=1, alpha=0.7, label=f'Level path {i+1}' if i == 0 else "")
    #         plt.plot(path_x, path_y, 'gx', markersize=4, alpha=0.7)

    # plt.axis('equal')
    # plt.title('Corpus Callosum Contour with Levelsets')
    # plt.legend(loc='best')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()

    return outside_contour, anterior_endpoint_idx, posterior_endpoint_idx
