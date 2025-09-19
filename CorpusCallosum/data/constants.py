from pathlib import Path

### Constants
WEIGHTS_PATH = Path(__file__).parent.parent / "weights"
FSAVERAGE_CENTROIDS_PATH = Path(__file__).parent / "fsaverage_centroids.json"
FSAVERAGE_DATA_PATH = Path(__file__).parent / "fsaverage_data.json"  # Contains both affine and header
FSAVERAGE_MIDDLE = 128  # Middle slice index in fsaverage space
CC_LABEL = 192          # Label value for corpus callosum in segmentation
FORNIX_LABEL = 250      # Label value for fornix in segmentation


STANDARD_OUTPUT_PATHS = {
    "upright_volume": "mri/upright_volume.mgz",
    "segmentation": "mri/cc_segmentation.mgz",
    "postproc_results": "stats/cc_postproc_results.json",
    "cc_markers": "stats/cc_markers.json",
    "upright_lta": "transforms/upright.lta",
    "orient_volume_lta": "transforms/orient_volume.lta",
    "orig_space_segmentation": "mri/segmentation_orig_space.mgz",
    "debug_image": "qc_snapshots/corpus_callosum.png",
    "thickness_image": "qc_snapshots/corpus_callosum_thickness.png"
}