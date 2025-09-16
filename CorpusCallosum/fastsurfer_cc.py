import argparse
import json
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from localization import localization_inference
from segmentation import segmentation_inference, segmentation_postprocessing
from recon_surf import lta
from CorpusCallosum.registration.mapping_helpers import interpolate_midplane, get_mapping_to_standard_space, map_softlabels_to_orig, apply_transform_to_pt, apply_transform_and_map_volume
from CorpusCallosum.shape.cc_postprocessing import process_slices, create_visualization

from FastSurferCNN.data_loader.conform import is_conform
from recon_surf.align_points import find_rigid
from CorpusCallosum.data.read_write import save_nifti_background, get_centroids_from_nib, convert_numpy_to_json_serializable, run_in_background, load_fsaverage_centroids, load_fsaverage_data

from CorpusCallosum.data.constants import *




def options_parse() -> argparse.Namespace:
    """Parse command line arguments for the pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_mri", type=str, required=False, help="Input MRI file path. If not provided, defaults to subject_dir/mri/orig.mgz")
    parser.add_argument("--aseg", type=str, required=False, help="Input segmentation file path. If not provided, defaults to subject_dir/mri/aparc.DKTatlas+aseg.deep.mgz")
    parser.add_argument("--subject_dir", type=str, required=False, help="Subject directory containing standard FreeSurfer structure. Required if --in_mri and --aseg are not both provided.", default=None)
    parser.add_argument("--debug_output_dir", type=str, required=False, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output and debug plots")
    
    # CC shape arguments
    parser.add_argument("--num_thickness_points", type=int, default=100, help="Number of points for thickness estimation.")
    parser.add_argument("--subdivisions", type=float, nargs='+', default=[1/6, 1/2, 2/3, 3/4], help="List of subdivision fractions for the corpus callosum subsegmentation.")
    parser.add_argument("--subdivision_method", type=str, default="shape", help="Method for contour subdivision. \
                        Options: shape (Intercallosal subdivision perpendicular to intercallosal line), vertical \
                        (orthogonal to the most anterior and posterior points in the AC/PC standardized CC contour), \
                        angular (subdivision based on equally spaced angles, as proposed by Hampel and colleagues), \
                        eigenvector (primary direction, same as FreeSurfers mri_cc)", choices=["shape", "vertical", "angular", "eigenvector"])
    parser.add_argument("--contour_smoothing", type=float, default=1.0, help="Gaussian sigma for smoothing during contour detection. Default is 1.0, higher values mean a smoother outline, at the cost of precision.")
    parser.add_argument("--slice_selection", type=str, default="middle", help="Which slices to process. Options: 'middle' (default), 'all', or a specific slice number.")
    
    # Output path arguments
    parser.add_argument("--upright_volume_path", type=str, help="Path for upright volume output (default: subject_dir/stats/upright_volume.mgz)", default=None)
    parser.add_argument("--segmentation_path", type=str, help="Path for segmentation output (default: subject_dir/stats/cc_segmentation.mgz)", default=None)
    parser.add_argument("--postproc_results_path", type=str, help="Path for postprocessing results (default: subject_dir/stats/cc_postproc_results.json)", default=None)
    parser.add_argument("--cc_markers_path", type=str, help="Path for CC markers output (default: subject_dir/stats/cc_markers.json)", default=None)
    parser.add_argument("--upright_lta_path", type=str, help="Path for upright LTA transform (default: subject_dir/transforms/upright.lta)", default=None)
    parser.add_argument("--orient_volume_lta_path", type=str, help="Path for orientation volume LTA transform (default: subject_dir/transforms/orient_volume.lta)", default=None)
    parser.add_argument("--orig_space_segmentation_path", type=str, help="Path for segmentation in original space (default: subject_dir/mri/segmentation_orig_space.mgz)", default=None)
    parser.add_argument("--debug_image_path", type=str, help="Path for debug visualization image (default: subject_dir/stats/cc_postprocessing.png)", default=None)
    
    # Template saving argument
    parser.add_argument("--save_template", type=str, help="Directory path where to save contours.txt and thickness_values.txt files", default=None)
    
    args = parser.parse_args()
    
    # Validation logic: either subject_dir OR both in_mri and aseg must be provided
    if not args.subject_dir and (not args.in_mri or not args.aseg):
        parser.error("You must specify either --subject_dir OR both --in_mri and --aseg arguments.")
    
    # If subject_dir is provided, set default paths for missing arguments
    if args.subject_dir:
        subject_dir_path = Path(args.subject_dir)
        
        # Create standard FreeSurfer subdirectories
        (subject_dir_path / "mri").mkdir(parents=True, exist_ok=True)
        (subject_dir_path / "stats").mkdir(parents=True, exist_ok=True)
        (subject_dir_path / "transforms").mkdir(parents=True, exist_ok=True)
        
        if not args.in_mri:
            args.in_mri = str(subject_dir_path / "mri" / "orig.mgz")
        
        if not args.aseg:
            args.aseg = str(subject_dir_path / "mri" / "aparc.DKTatlas+aseg.deep.mgz")
        
        # Set default output paths if not provided
        for key, value in STANDARD_OUTPUT_PATHS.items():
            if not getattr(args, f"{key}_path"):
                setattr(args, f"{key}_path", str(subject_dir_path / value))

        # Set output_dir to subject_dir
        args.output_dir = str(subject_dir_path)

    
    # Create parent directories for all output paths
    for path in [args.upright_volume_path, args.segmentation_path, args.postproc_results_path, args.cc_markers_path, args.upright_lta_path, args.orient_volume_lta_path]:
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
                
    return args


def centroid_registration(aseg_nib, verbose=False):
    """Perform centroid-based registration between subject and fsaverage space.
    
    Computes a rigid transformation between the subject's segmentation and fsaverage space
    by aligning centroids of corresponding anatomical structures.
    
    Args:
        aseg_nib (nib.Nifti1Image): Subject's segmentation image
        verbose (bool): Whether to print progress information
        
    Returns:
        tuple: Contains:
            - orig_fsaverage_vox2vox: Transformation matrix from original to fsaverage voxel space
            - orig_fsaverage_ras2ras: Transformation matrix from original to fsaverage RAS space
            - fsaverage_hires_affine: High-resolution fsaverage affine matrix
            - fsaverage_header: FSAverage header fields for LTA writing
    """
    if verbose:
        print("Centroid registration")

    # Load pre-computed fsaverage centroids and data from static files
    centroids_dst = load_fsaverage_centroids(FSAVERAGE_CENTROIDS_PATH)
    fsaverage_affine, fsaverage_header = load_fsaverage_data(FSAVERAGE_DATA_PATH)
    
    centroids_mov, ids_not_found = get_centroids_from_nib(aseg_nib, label_ids=list(centroids_dst.keys()))

    # delete not found labels from centroids_mov
    for id in ids_not_found:
        del centroids_dst[id]

    centroids_mov = np.array(list(centroids_mov.values())).T
    centroids_dst = np.array(list(centroids_dst.values())).T

    orig_fsaverage_ras2ras = find_rigid(p_mov=centroids_mov.T, p_dst=centroids_dst.T)

    # make affine that increases resolution to orig resolution
    resolution_orig = aseg_nib.header.get_zooms()[0]
    resolution_trans = np.eye(4)
    resolution_trans[0, 0] = resolution_orig
    resolution_trans[1, 1] = resolution_orig
    resolution_trans[2, 2] = resolution_orig

    orig_fsaverage_vox2vox = np.linalg.inv(resolution_trans @ fsaverage_affine) @ orig_fsaverage_ras2ras @ aseg_nib.affine
    fsaverage_hires_affine = resolution_trans @ fsaverage_affine

    return orig_fsaverage_vox2vox, orig_fsaverage_ras2ras, fsaverage_hires_affine, fsaverage_header


def localize_ac_pc(midslices, aseg_nib, orig_fsaverage_vox2vox, model_localization, slices_to_analyze, verbose=False):
    """Localize anterior and posterior commissure points in the brain.
    
    Uses a trained model to detect AC and PC points in mid-sagittal slices,
    using the third ventricle as an anatomical reference.
    
    Args:
        midslices (np.ndarray): Array of mid-sagittal slices
        aseg_nib (nib.Nifti1Image): Subject's segmentation image
        orig_fsaverage_vox2vox (np.ndarray): Transformation matrix to fsaverage space
        fsaverage_hires_affine (np.ndarray): High-resolution fsaverage affine matrix
        model_localization: Trained model for AC-PC detection
        slices_to_analyze (int): Number of slices to process
        verbose (bool): Whether to print progress information
        
    Returns:
        tuple: Contains:
            - ac_coords (np.ndarray): Coordinates of the anterior commissure
            - pc_coords (np.ndarray): Coordinates of the posterior commissure
    """
    if verbose:
        print("Localization and segmentation inference")

    # get center of third ventricle from aseg and map to fsaverage space
    third_ventricle_mask = aseg_nib.get_fdata() == 4
    third_ventricle_center = np.argwhere(third_ventricle_mask).mean(axis=0)
    third_ventricle_center_vox = apply_transform_to_pt(third_ventricle_center, orig_fsaverage_vox2vox, inv=False)

    # get 5 mm of slices output with 3 slices per inference
    midslices_middle = midslices.shape[0] // 2
    middle_slices_localization = midslices[midslices_middle-slices_to_analyze//2-1:midslices_middle+slices_to_analyze//2+2] 
    ac_coords, pc_coords = localization_inference.run_inference_on_slice(model_localization, middle_slices_localization, third_ventricle_center_vox[1:])

    return ac_coords, pc_coords


def segment_cc(midslices, ac_coords, pc_coords, aseg_nib, model_segmentation, slices_to_analyze):
    """Segment the corpus callosum using a trained model.
    
    Performs corpus callosum segmentation on mid-sagittal slices using a trained model,
    with AC-PC points as anatomical references. Includes post-processing to clean the segmentation.
    
    Args:
        midslices (np.ndarray): Array of mid-sagittal slices
        ac_coords (np.ndarray): Anterior commissure coordinates
        pc_coords (np.ndarray): Posterior commissure coordinates
        aseg_nib (nib.Nifti1Image): Subject's segmentation image
        orig_fsaverage_vox2vox (np.ndarray): Transformation matrix to fsaverage space
        fsaverage_hires_affine (np.ndarray): High-resolution fsaverage affine matrix
        model_segmentation: Trained model for CC segmentation
        slices_to_analyze (int): Number of slices to process
        verbose (bool): Whether to print progress information
        
    Returns:
        tuple: Contains:
            - segmentation (np.ndarray): Binary segmentation of the corpus callosum
            - outputs_soft (np.ndarray): Soft segmentation probabilities
    """
    # get 5 mm of slices output with 9 slices per inference
    midslices_middle = midslices.shape[0] // 2
    middle_slices_segmentation = midslices[midslices_middle-slices_to_analyze//2-4:midslices_middle+slices_to_analyze//2+5]
    segmentation, inputs, outputs_avg, outputs_soft = segmentation_inference.run_inference_on_slice(model_segmentation, 
                                                                                                    middle_slices_segmentation, 
                                                                                                    AC_center=ac_coords, PC_center=pc_coords, 
                                                                                                    voxel_size=aseg_nib.header.get_zooms()[0])    

    pre_clean_segmentation = segmentation.copy()
    segmentation, cc_volume_mask = segmentation_postprocessing.clean_cc_segmentation(segmentation)

    # print a warning if the cc_volume_mask touches the edge of the segmentation
    if np.any(cc_volume_mask[:,0,:]) or np.any(cc_volume_mask[:,-1,:]) or np.any(cc_volume_mask[:,:,0]) or np.any(cc_volume_mask[:,:,-1]):
        print("Warning: CC volume mask touches the edge of the segmentation field-of-view, CC might be truncated")

    # get voxels that were removed during cleaning
    removed_voxels = pre_clean_segmentation != segmentation
    outputs_soft[removed_voxels, 1] = 0

    return segmentation, outputs_soft


def main(in_mri_path: str | Path, aseg_path: str | Path, output_dir: str | Path, slice_selection: str = "middle", 
         debug_output_dir: str | Path = None, verbose: bool = False, num_thickness_points: int = 100,
         subdivisions: list[float] | None = None, subdivision_method: str = "shape", 
         contour_smoothing: float = 1.0,
         upright_volume_path: str | Path = None, segmentation_path: str | Path = None,
         postproc_results_path: str | Path = None, cc_markers_path: str | Path = None,
         upright_lta_path: str | Path = None, orient_volume_lta_path: str | Path = None,
         orig_space_segmentation_path: str | Path = None, debug_image_path: str | Path = None,
         save_template: str | Path | None = None) -> None:
    """Main pipeline function for corpus callosum analysis.
    
    This function performs the following steps:
    1. Initializes environment and loads models
    2. Registers input image to fsaverage space
    3. Detects AC and PC points
    4. Segments the corpus callosum
    5. Performs enhanced post-processing analysis
    6. Saves results and visualizations
    
    Args:
        in_mri_path: Path to input MRI file
        aseg_path: Path to input segmentation file
        output_dir: Directory for output files
        slice_selection: Which slices to process ('middle', 'all', or specific slice number)
        debug_output_dir: Optional directory for debug outputs
        verbose: Flag for verbose output
        num_thickness_points: Number of points for thickness estimation
        subdivisions: List of subdivision fractions for CC subsegmentation
        subdivision_method: Method for contour subdivision
        contour_smoothing: Gaussian sigma for smoothing during contour detection
        upright_volume_path: Path for upright volume output (default: output_dir/upright_volume.mgz)
        segmentation_path: Path for segmentation output (default: output_dir/segmentation.mgz)
        postproc_results_path: Path for postprocessing results (default: output_dir/cc_postproc_results.json)
        cc_markers_path: Path for CC markers output (default: output_dir/cc_markers.json)
        upright_lta_path: Path for upright LTA transform (default: output_dir/upright.lta)
        orient_volume_lta_path: Path for orientation volume LTA transform (default: output_dir/orient_volume.lta)
        orig_space_segmentation_path: Path for segmentation in original space (default: output_dir/mri/segmentation_orig_space.mgz)
        debug_image_path: Path for debug visualization image (default: output_dir/stats/cc_postprocessing.png)
        save_template: Directory path where to save contours.txt and thickness_values.txt files
        
    The function saves multiple outputs to specified paths or default locations in output_dir:
    - cc_markers.json: Contains detected landmarks and measurements
    - midplane_slices.mgz: Extracted midplane slices
    - upright_volume.mgz: Volume aligned to standard orientation
    - segmentation.mgz: Corpus callosum segmentation
    - cc_postproc_results.json: Enhanced postprocessing results
    - Various visualization plots and transformation matrices
    """
    
    if subdivisions is None:
        subdivisions = [1/6, 1/2, 2/3, 3/4]
    
    # Convert all paths to Path objects
    in_mri_path = Path(in_mri_path)
    aseg_path = Path(aseg_path)
    output_dir = Path(output_dir)
    debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
    save_template = Path(save_template) if save_template else None
    
    # Validate subdivision fractions
    for i in subdivisions:
        if i < 0 or i > 1:
            print('Error: Subdivision fractions must be between 0 and 1, but got: ', i)
            exit(1)

    #### setup variables
    IO_processes = []
    
    orig = nib.load(in_mri_path)
    

    # 5 mm around the midplane
    slices_to_analyze = int(np.ceil(5 / orig.header.get_zooms()[0]))
    if slices_to_analyze % 2 == 0:
        slices_to_analyze += 1

    if verbose:
        print(f"Segmenting {slices_to_analyze} slices (5 mm width at {orig.header.get_zooms()[0]} mm resolution, center around the mid-sagittal plane)")


    if not is_conform(orig, conform_vox_size=orig.header.get_zooms()[0]):
        print("Error: MRI is not conformed, please run conform.py or mri_convert to conform the image.")
        exit(1)

    # load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_localization = localization_inference.load_model(str(Path(WEIGHTS_PATH) / "localization_weights_acpc.pth"), device=device)
    model_segmentation = segmentation_inference.load_model(str(Path(WEIGHTS_PATH) / "segmentation_weights_cc_fn.pth"), device=device)


    aseg_nib = nib.load(aseg_path)

    orig_fsaverage_vox2vox, orig_fsaverage_ras2ras, fsaverage_hires_affine, fsaverage_header = centroid_registration(aseg_nib, verbose)

    if verbose:
        print("Interpolating midplane")
    
    # this is a fast interpolation to not block the main thread
    midslices = interpolate_midplane(orig, orig_fsaverage_vox2vox, slices_to_analyze)


    # start saving upright volume
    IO_processes.append(run_in_background(apply_transform_and_map_volume, False, 
                                          orig.get_fdata(), orig_fsaverage_vox2vox, fsaverage_hires_affine, None, upright_volume_path, output_size=np.array([256,256,256])))
    
    #### do localization and segmentation inference
    ac_coords, pc_coords = localize_ac_pc(midslices, aseg_nib, orig_fsaverage_vox2vox, model_localization, slices_to_analyze, verbose)
    segmentation, outputs_soft = segment_cc(midslices, ac_coords, pc_coords, aseg_nib, model_segmentation, slices_to_analyze)
    
    # map soft labels to original space (in parallel because this takes a while)
    IO_processes.append(run_in_background(map_softlabels_to_orig, False, 
                                        outputs_soft, orig_fsaverage_vox2vox, orig, slices_to_analyze, orig_space_segmentation_path, fsaverage_middle=FSAVERAGE_MIDDLE))

    # Create a temporary segmentation image with proper affine for enhanced postprocessing
    temp_seg_affine = fsaverage_hires_affine @ np.linalg.inv(np.eye(4))
    
    # Process slices based on selection mode
    slice_results, slice_io_processes = process_slices(
        segmentation=segmentation,
        slice_selection=slice_selection,
        temp_seg_affine=temp_seg_affine,
        midslices=midslices,
        ac_coords=ac_coords,
        pc_coords=pc_coords,
        num_thickness_points=num_thickness_points,
        subdivisions=subdivisions,
        subdivision_method=subdivision_method,
        contour_smoothing=contour_smoothing,
        output_dir=output_dir,
        debug_image_path=debug_image_path,
        vox_size=orig.header.get_zooms()[0],
        verbose=verbose,
        save_template=save_template
    )
    IO_processes.extend(slice_io_processes)
    
    # Get middle slice result for backward compatibility
    middle_slice_result = slice_results[len(slice_results)//2]
    
    # Create enhanced output dictionary with all slice results
    per_slice_output_dict = {
        'slices': [convert_numpy_to_json_serializable({
            'slice_index': result['slice_index'],
            'cc_index': result['cc_index'],
            'circularity': result['circularity'],
            'areas': result['areas'],
            'midline_length': result['midline_length'],
            'thickness': result['thickness'],
            'curvature': result['curvature'],
            'thickness_profile': result['thickness_profile'],
            'total_area': result['total_area'],
            'total_perimeter': result['total_perimeter']
        }) for result in slice_results],
        'slices_in_segmentation': segmentation.shape[0],
        'voxel_size': [float(x) for x in orig.header.get_zooms()],
        'subdivision_method': subdivision_method,
        'num_thickness_points': num_thickness_points,
        'subdivisions': subdivisions,
        'contour_smoothing': contour_smoothing,
        'slice_selection': slice_selection
    }

    # Save slice-wise postprocessing results to JSON
    with open(postproc_results_path, "w") as f:
        json.dump(per_slice_output_dict, f, indent=4)

    if verbose:
        print(f"Multiple slice post-processing results saved to {postproc_results_path}")
    
    ########## Save outputs ##########

    cc_volume = segmentation_postprocessing.get_cc_volume(desired_width_mm=5, cc_mask=segmentation == CC_LABEL, voxel_size=orig.header.get_zooms())

    # Create backward compatible output_dict for existing pipeline using middle slice
    output_dict = {
        'areas': middle_slice_result['areas'],
        'areas_hofer_frahm': middle_slice_result['areas'] if middle_slice_result['split_contours_hofer_frahm'] is not None else middle_slice_result['areas'],
        'thickness': middle_slice_result['thickness'],
        'curvature': middle_slice_result['curvature'],
        'midline_length': middle_slice_result['midline_length'],
        'circularity': middle_slice_result['circularity'],
        'cc_index': middle_slice_result['cc_index'],
        'total_area': middle_slice_result['total_area'],
        'total_perimeter': middle_slice_result['total_perimeter'],
        'thickness_profile': middle_slice_result['thickness_profile']
    }
    
    # multiply split contour with resolution scale factor for middle slice visualization
    split_contours = [split_contour * orig.header.get_zooms()[1] for split_contour in middle_slice_result['split_contours']]
    if middle_slice_result['split_contours_hofer_frahm'] is not None:
        split_contours_hofer_frahm = [split_contour * orig.header.get_zooms()[1] for split_contour in middle_slice_result['split_contours_hofer_frahm']]
    else:
        split_contours_hofer_frahm = split_contours  # backward compatibility
    midline_equidistant = middle_slice_result['midline_equidistant'] * orig.header.get_zooms()[1]
    levelpaths = [levelpath * orig.header.get_zooms()[1] for levelpath in middle_slice_result['levelpaths']]
    
    # Save middle slice visualization
    single_slice_result = {
        'split_contours': split_contours,
        'split_contours_hofer_frahm': split_contours_hofer_frahm,
        'midline_equidistant': midline_equidistant,
        'levelpaths': levelpaths
    }
    IO_processes.append(create_visualization(subdivision_method, single_slice_result, midslices, 
                                           output_dir, ac_coords, pc_coords, orig.header.get_zooms()[0], ' (Middle Slice)'))
    
    # get ac and pc in all spaces
    ac_coords_3d = np.hstack((FSAVERAGE_MIDDLE, ac_coords))
    pc_coords_3d = np.hstack((FSAVERAGE_MIDDLE, pc_coords))
    standardized_to_orig_vox2vox, ac_coords_standardized, pc_coords_standardized, ac_coords_orig, pc_coords_orig = get_mapping_to_standard_space(orig, ac_coords_3d, pc_coords_3d, orig_fsaverage_vox2vox, output_dir)


    # save segmentation with fitting affine
    orig_to_seg = np.eye(4)
    orig_to_seg[0, 3] = -FSAVERAGE_MIDDLE+slices_to_analyze//2
    seg_affine = fsaverage_hires_affine
    seg_affine = seg_affine @ np.linalg.inv(orig_to_seg)
    save_nifti_background(IO_processes, segmentation, seg_affine, orig.header, segmentation_path)

    # write output dict as csv
    output_dict["ac_center"] = ac_coords_orig
    output_dict["pc_center"] = pc_coords_orig
    output_dict["ac_center_oriented_volume"] = ac_coords_standardized
    output_dict["pc_center_oriented_volume"] = pc_coords_standardized
    output_dict["ac_center_upright"] = ac_coords_3d
    output_dict["pc_center_upright"] = pc_coords_3d
    output_dict["cc_5mm_volume"] = cc_volume
    output_dict["num_slices"] = slices_to_analyze

    # Convert numpy arrays to lists for JSON serialization
    output_dict = convert_numpy_to_json_serializable(output_dict)

    with open(cc_markers_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    # save lta to fsaverage space
    lta.writeLTA(upright_lta_path, orig_fsaverage_ras2ras, aseg_path, aseg_nib.header, 'fsaverage', fsaverage_header)

    # save lta to standardized space (fsaverage + nodding + ac to center)
    orig_to_standardized_ras2ras = orig.affine @ np.linalg.inv(standardized_to_orig_vox2vox) @ np.linalg.inv(orig.affine)
    lta.writeLTA(orient_volume_lta_path, orig_to_standardized_ras2ras, in_mri_path, orig.header, in_mri_path, orig.header)

    for process in IO_processes:
        if process is not None:
            process.join()


if __name__ == "__main__":
    options = options_parse()
    main_args = vars(options)
    
    # Rename keys to match main function parameters
    main_args['in_mri_path'] = main_args.pop('in_mri')
    main_args['aseg_path'] = main_args.pop('aseg')
    main_args['output_dir'] = main_args.pop('subject_dir', '.')
    
    main(**main_args)
