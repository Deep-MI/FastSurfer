from pathlib import Path

### Constants
WEIGHTS_PATH = Path(__file__).parent.parent / "weights"
FSAVERAGE_CENTROIDS_PATH = Path(__file__).parent / "fsaverage_centroids.json"
FSAVERAGE_DATA_PATH = Path(__file__).parent / "fsaverage_data.json"  # Contains both affine and header
FSAVERAGE_MIDDLE = 128  # Middle slice index in fsaverage space
CC_LABEL = 192          # Label value for corpus callosum in segmentation
FORNIX_LABEL = 250      # Label value for fornix in segmentation
SUBSEGEMNT_LABELS = [251, 252, 253, 254, 255] # labels for subsegments in segmentation


STANDARD_OUTPUT_PATHS = {
    ## images
    "upright_volume": None, # orig.mgz mapped to upright space
    ## segmentations
    "segmentation": "mri/callosum_seg_upright.mgz", # corpus callosum segmentation in upright space
    "orig_space_segmentation": "mri/callosum_seg_aseg_space.mgz", # cc segmentation in input segmentations space
    "softlabels_cc": "mri/callosum_seg_soft.mgz", # cc softlabels  in upright space
    "softlabels_fn": "mri/fornix_seg_soft.mgz", # fornix softlabels in upright space
    "softlabels_background": "mri/background_seg_soft.mgz", # background softlabels in upright space
    ## stats
    "cc_markers": "stats/callosum.CC.midslice.json", # cc metrics for middle slice
    "postproc_results": "stats/callosum.CC.all_slices.json", # cc metrics for all slices
    ## transforms
    "upright_lta": "mri/transforms/cc_up.lta", # lta transform from orig to upright space
    "orient_volume_lta": "mri/transforms/orient_volume.lta", # lta transform from orig to upright+acpc corrected space
    ## qc
    "debug_image": "qc_snapshots/callosum.png", # debug image of cc contours
    "thickness_image": "qc_snapshots/callosum_thickness.png", # whippersnappy 3D image of cc thickness
    "cc_html": "qc_snapshots/corpus_callosum.html", # plotly cc visualization
    ## surface
    "surf_file": "surf/callosum.surf", # cc surface file
    "overlay_file": "surf/callosum.thickness.w", # cc surface overlay file
    "vtk_file": "qc_snapshots/callosum_mesh.vtk", # vtk file of cc mesh
}