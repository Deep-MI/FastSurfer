import multiprocessing

import nibabel as nib
import numpy as np

import FastSurferCNN.utils.logging as logging

logger = logging.get_logger(__name__)


def run_in_background(function, debug=False, *args, **kwargs):
    """Run a function in the background using multiprocessing.
    
    This function executes the given function either in a separate process (normal mode)
    or in the current process (debug mode). In debug mode, the function is executed
    synchronously for easier debugging.
    
    Args:
        function: The function to execute
        debug (bool): If True, run synchronously in current process
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        multiprocessing.Process or None: Process object if running in background,
            None if in debug mode
    """
    if debug:
        function(*args, **kwargs)
        process = None
    else:
        process = multiprocessing.Process(target=function, args=args, kwargs=kwargs)
        process.start()
    return process



def get_centroids_from_nib(seg_img: nib.Nifti1Image, label_ids: list[int] | None = None) -> dict[int, np.ndarray]:
    """Get centroids of segmentation labels in RAS coordinates.
    
    Calculates the centroid coordinates for each segmentation label in the image.
    If label_ids is provided, only calculates centroids for those specific labels.
    Coordinates are returned in RAS (Right-Anterior-Superior) coordinate system.
    
    Args:
        seg_img (nib.Nifti1Image): Nibabel image containing segmentation labels
        label_ids (list[int] | None): Optional list of specific label IDs to process.
            If None, processes all non-zero labels.
        
    Returns:
        If label_ids is None:
            dict[int, np.ndarray]: Mapping of label IDs to their centroids (x,y,z) in RAS coordinates
        If label_ids is provided:
            tuple: Contains:
                - dict[int, np.ndarray]: Mapping of found label IDs to their centroids
                - list[int]: List of label IDs that were not found in the image
    """
    # Get segmentation data and affine
    seg_data = seg_img.get_fdata()
    vox2ras = seg_img.affine
    
    # Get unique labels
    if label_ids is None:
        labels = np.unique(seg_data)
        labels = labels[labels > 0]  # Exclude background
    else:
        labels = label_ids
    
    centroids = {}
    ids_not_found = []
    for label in labels:
        # Get voxel indices for this label
        vox_coords = np.array(np.where(seg_data == label))
        if vox_coords.size == 0:
            ids_not_found.append(label)
            continue
        # Calculate centroid in voxel space
        vox_centroid = np.mean(vox_coords, axis=1)
        
        # Convert to homogeneous coordinates
        vox_centroid = np.append(vox_centroid, 1)
        
        # Transform to RAS coordinates
        ras_centroid = vox2ras @ vox_centroid
        
        # Store without homogeneous coordinate
        centroids[int(label)] = ras_centroid[:3]
        
    if label_ids is not None:
        return centroids, ids_not_found
    else:
        return centroids



def save_nifti_background(io_processes, data, affine, header, filepath):
    """Save a NIfTI image in a background process.
    
    Creates a MGHImage from the provided data and metadata, then saves it to disk
    using a background process to avoid blocking the main execution.
    
    Args:
        io_processes (list): List to store background process handles
        data (np.ndarray): Image data array
        affine (np.ndarray): 4x4 affine transformation matrix
        header: NIfTI header object containing metadata
        filepath (str): Path where the image should be saved
    """
    logger.info(f"Saving NIfTI image to {filepath}")
    io_processes.append(run_in_background(nib.save, False, 
                                        nib.MGHImage(data, affine, header), filepath))


def convert_numpy_to_json_serializable(obj):
    """Convert numpy arrays in nested data structures to JSON serializable format.
    
    Recursively traverses dictionaries, lists, and numpy arrays, converting numpy arrays
    to Python lists and numpy scalars to Python scalars for JSON serialization.
    
    Args:
        obj: Any Python object that may contain numpy arrays (dict, list, np.ndarray, or scalar)
        
    Returns:
        The input object with all numpy arrays converted to lists and numpy scalars to Python scalars
        
    Example:
        >>> data = {'array': np.array([1, 2, 3]), 'nested': {'array': np.array([4, 5])}}
        >>> result = convert_numpy_to_json_serializable(data)
        >>> # Result: {'array': [1, 2, 3], 'nested': {'array': [4, 5]}}
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


def load_fsaverage_centroids(centroids_path):
    """Load fsaverage centroids from static JSON file.
    
    Loads pre-computed centroids from a static JSON file, avoiding the need to
    compute them from the fsaverage segmentation at runtime.
    
    Args:
        centroids_path (str or Path): Path to the JSON file containing centroids
        
    Returns:
        dict[int, np.ndarray]: Mapping of label IDs to their centroids (x,y,z) in RAS coordinates
        
    Raises:
        FileNotFoundError: If the centroids file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    import json
    from pathlib import Path
    
    centroids_path = Path(centroids_path)
    if not centroids_path.exists():
        raise FileNotFoundError(f"Fsaverage centroids file not found: {centroids_path}")
    
    with open(centroids_path) as f:
        centroids_data = json.load(f)
    
    # Convert string keys back to integers and lists back to numpy arrays
    centroids = {}
    for label_str, centroid_list in centroids_data.items():
        label_id = int(label_str)
        centroids[label_id] = np.array(centroid_list)
    
    return centroids


def load_fsaverage_affine(affine_path):
    """Load fsaverage affine matrix from static text file.
    
    Loads pre-computed affine matrix from a static text file, avoiding the need to
    load the fsaverage segmentation at runtime.
    
    Args:
        affine_path (str or Path): Path to the text file containing affine matrix
        
    Returns:
        np.ndarray: 4x4 affine transformation matrix
        
    Raises:
        FileNotFoundError: If the affine file doesn't exist
        ValueError: If the file doesn't contain a valid 4x4 matrix
    """
    from pathlib import Path
    
    affine_path = Path(affine_path)
    if not affine_path.exists():
        raise FileNotFoundError(f"Fsaverage affine file not found: {affine_path}")
    
    affine_matrix = np.loadtxt(affine_path)
    
    if affine_matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine matrix, got shape {affine_matrix.shape}")
    
    return affine_matrix


def load_fsaverage_data(data_path):
    """Load fsaverage affine matrix and header fields from static JSON file.
    
    Loads pre-computed affine matrix and header fields from a static JSON file,
    avoiding the need to load the fsaverage segmentation at runtime.
    
    Args:
        data_path (str or Path): Path to the JSON file containing combined data
        
    Returns:
        tuple: Contains:
            - affine_matrix (np.ndarray): 4x4 affine transformation matrix
            - header_fields (dict): Header fields needed for LTA:
                - dims (list[int]): Volume dimensions [x,y,z]
                - delta (list[float]): Voxel size in mm [x,y,z]
                - Mdc (np.ndarray): 3x3 direction cosines matrix
                - Pxyz_c (np.ndarray): RAS center coordinates [x,y,z]
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        ValueError: If required fields are missing
    """
    import json
    from pathlib import Path
    
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
    
    header_fields = ["dims", "delta", "Mdc", "Pxyz_c"]
    for field in header_fields:
        if field not in data["header"]:
            raise ValueError(f"Required header field missing: {field}")
    
    # Convert lists back to numpy arrays
    affine_matrix = np.array(data["affine"])
    header_data = data["header"].copy()
    header_data["Mdc"] = np.array(header_data["Mdc"])
    header_data["Pxyz_c"] = np.array(header_data["Pxyz_c"])
    
    # Validate affine matrix shape
    if affine_matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine matrix, got shape {affine_matrix.shape}")
    
    return affine_matrix, header_data