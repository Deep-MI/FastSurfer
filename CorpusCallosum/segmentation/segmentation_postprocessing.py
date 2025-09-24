import numpy as np
from scipy import integrate, ndimage
from skimage.measure import label

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import CC_LABEL, FORNIX_LABEL

logger = logging.get_logger(__name__)




def get_cc_volume(desired_width_mm: int, cc_mask: np.ndarray, voxel_size: tuple[float, float, float]) -> float:
    """Calculate the volume of the corpus callosum in cubic millimeters.
    
    This function calculates the volume of the corpus callosum (CC) in cubic millimeters.
    If the CC width is larger than desired_width_mm, the voxels on the edges are calculated as
    partial volumes to achieve the desired width.
    
    Args:
        desired_width_mm (int): Desired width of the CC in millimeters
        cc_mask (np.ndarray): Binary mask of the corpus callosum
        voxel_size (tuple[float, float, float]): Voxel size in millimeters (x, y, z)
        
    Returns:
        float: Volume of the CC in cubic millimeters
        
    Raises:
        ValueError: If CC width is smaller than desired width
        AssertionError: If CC mask doesn't have odd number of voxels in x dimension
    """
    assert cc_mask.shape[0] % 2 == 1, "CC mask must have odd number of voxels in x dimension"


    # Calculate voxel volume
    voxel_volume = np.prod(voxel_size)

    # Get width of CC mask in voxels by finding the extent in x dimension
    width_vox = np.sum(np.any(cc_mask, axis=(1,2)))

    # we are in LIA, so 0 is L/R resolution
    width_mm = width_vox * voxel_size[0]

    if width_mm == desired_width_mm:
        return np.sum(cc_mask) * voxel_volume
    elif width_mm > desired_width_mm:
        # remainder on the left/right side of the CC mask
        desired_width_vox = desired_width_mm / voxel_size[0]
        fraction_of_voxel_at_edge = (desired_width_vox % 1) / 2

        if fraction_of_voxel_at_edge > 0:
            desired_width_vox = int(np.floor(desired_width_vox) + 1)
            desired_width_vox = desired_width_vox + 1 if desired_width_vox % 2 == 0 else desired_width_vox

            assert cc_mask.shape[0] == desired_width_vox, (f"CC mask should have {desired_width_vox} voxels, "
                                                          f"but has {cc_mask.shape[0]}")

        left_partial_volume = np.sum(cc_mask[0]) * voxel_volume * fraction_of_voxel_at_edge
        right_partial_volume = np.sum(cc_mask[-1]) * voxel_volume * fraction_of_voxel_at_edge
        center_volume = np.sum(cc_mask[1:-1]) * voxel_volume
        return left_partial_volume + right_partial_volume + center_volume
    else:
        raise ValueError(f"Width of CC segmentation is smaller than desired width: {width_mm} < {desired_width_mm}")

def get_cc_volume_simpsons(desired_width_mm: int, cc_contours: list[np.ndarray], 
                           voxel_size: tuple[float, float, float]) -> float:
    """Calculate the volume of the corpus callosum in cubic millimeters using Simpson's rule.
    
    This function calculates the volume of the corpus callosum (CC) in cubic millimeters using Simpson's rule.
    If the CC width is larger than desired_width_mm, the voxels on the edges are calculated as
    partial volumes to achieve the desired width.
    
    Args:
        desired_width_mm (int): Desired width of the CC in millimeters
        cc_contours (list[np.ndarray]): List of CC contours for each slice in the left-right direction
        voxel_size (tuple[float, float, float]): Voxel size in millimeters (x, y, z)
        
    Returns:
        float: Volume of the CC in cubic millimeters
        
    Raises:
        ValueError: If CC width is smaller than desired width or insufficient contours for Simpson's rule
    """
    if len(cc_contours) < 3:
        raise ValueError("Need at least 3 contours for Simpson's rule integration")
    
    # Calculate cross-sectional areas for each contour
    areas = []
    for contour in cc_contours:
        # Calculate area using the shoelace formula for polygon area
        if contour.shape[1] < 3:
            areas.append(0.0)
        else:
            x = contour[0]
            y = contour[1]
            # Shoelace formula: A = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
            area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
            # Convert from voxel^2 to mm^2
            area_mm2 = area * voxel_size[1] * voxel_size[2]  # y * z voxel dimensions
            areas.append(area_mm2)
    
    areas = np.array(areas)
    
    # Calculate spacing between slices (left-right direction)
    lr_spacing = voxel_size[0]  # x-direction voxel size
    
    # Get current width in mm
    current_width_mm = len(cc_contours) * lr_spacing
    
    if current_width_mm == desired_width_mm:
        # Use Simpson's rule directly
        return integrate.simpson(areas, dx=lr_spacing)
    elif current_width_mm > desired_width_mm:
        # Handle partial volumes at edges
        desired_width_vox = desired_width_mm / lr_spacing
        fraction_of_voxel_at_edge = (desired_width_vox % 1) / 2
        
        if fraction_of_voxel_at_edge > 0:
            # Apply partial volume correction to edge areas
            areas_corrected = areas.copy()
            areas_corrected[0] *= fraction_of_voxel_at_edge
            areas_corrected[-1] *= fraction_of_voxel_at_edge
            
            # Use Simpson's rule with corrected areas
            return integrate.simps(areas_corrected, dx=lr_spacing)
        else:
            # No partial volumes needed, truncate to desired width
            desired_slices = int(desired_width_vox)
            if desired_slices % 2 == 0:
                desired_slices += 1  # Ensure odd number for Simpson's rule
            
            start_idx = (len(areas) - desired_slices) // 2
            end_idx = start_idx + desired_slices
            truncated_areas = areas[start_idx:end_idx]
            
            return integrate.simps(truncated_areas, dx=lr_spacing)
    else:
        raise ValueError(f"Width of CC segmentation is smaller than desired width: {current_width_mm} < {desired_width_mm}")

def get_largest_cc(seg_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get largest connected component from a binary segmentation array.
    
    This function takes a binary segmentation array, dilates it, finds connected components,
    and returns the largest component (excluding background) along with its mask.
    
    Args:
        seg_arr (np.ndarray): Input binary segmentation array
        
    Returns:
        tuple: A tuple containing:
            - clean_seg (np.ndarray): Segmentation array with only the largest connected component
            - largest_cc (np.ndarray): Binary mask of the largest connected component
    """
    # generate dilatation structure
    struct1 = ndimage.generate_binary_structure(3, 3)
    # Dilate prediction
    mask = ndimage.binary_dilation(seg_arr, structure=struct1, iterations=1, ).astype(np.uint8)
    # Get connected components
    labels_cc = label(mask, connectivity=3, background=0)
    # Get componnets count
    bincount = np.bincount(labels_cc.flat)
    # Get background label, assumption that background is the biggest connected component
    background = np.argmax(bincount)
    bincount[background] = -1
    # Get largest connected component
    largest_cc = labels_cc == np.argmax(bincount)
    # Apply mask
    clean_seg = seg_arr * largest_cc

    return clean_seg,largest_cc

def clean_cc_segmentation(seg_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clean corpus callosum segmentation by removing non-connected components.
    
    This function processes a segmentation array to clean up the corpus callosum (CC)
    by removing non-connected components. It first isolates the CC (label 192),
    removes non-connected components, then adds the fornix (label 250), and
    finally removes non-connected components from the combined CC and fornix.
    
    Args:
        seg_arr (np.ndarray): Input segmentation array with CC (192) and fornix (250) labels
        
    Returns:
        tuple: A tuple containing:
            - clean_seg (np.ndarray): Cleaned segmentation array with only the largest 
              connected component of CC and fornix
            - mask (np.ndarray): Binary mask of the largest connected component
    """
    #Remove non connected components from the CC alone
    clean_seg = np.zeros_like(seg_arr)
    clean_seg[seg_arr == CC_LABEL] = CC_LABEL
    clean_seg,_ = get_largest_cc(clean_seg)

    #Add fornix to the CC labels
    clean_seg[seg_arr == FORNIX_LABEL] = FORNIX_LABEL

    #Remove non connected components from CC & Fornix
    clean_seg, mask = get_largest_cc(clean_seg)

    unique_labels = np.unique(clean_seg)

    if 250 not in unique_labels:
        clean_seg[seg_arr == 250] = 250
        mask [seg_arr == 250] = True
    if 192 not in unique_labels:
        clean_seg[seg_arr == 192] = 192
        mask[seg_arr == 192] = True
    return clean_seg, mask
