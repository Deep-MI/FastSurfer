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


from pathlib import Path

### Constants
WEIGHTS_PATH = Path(__file__).parent.parent.parent / "checkpoints"
FSAVERAGE_CENTROIDS_PATH = Path(__file__).parent / "fsaverage_centroids.json"
FSAVERAGE_DATA_PATH = Path(__file__).parent / "fsaverage_data.json"  # Contains both affine and header
FSAVERAGE_MIDDLE = 128  # Middle slice index in fsaverage space
CC_LABEL = 192          # Label value for corpus callosum in segmentation
FORNIX_LABEL = 250      # Label value for fornix in segmentation
SUBSEGMENT_LABELS = [251, 252, 253, 254, 255] # labels for subsegments in segmentation
FASTSURFER_ROOT = Path(__file__).parent.parent.parent # TODO: use FastSurfer function for this


STANDARD_INPUT_PATHS = {
    "t1": "mri/orig.mgz",
    "aseg_name": "mri/aparc.DKTatlas+aseg.deep.mgz",
}

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
    "qc_image": "qc_snapshots/callosum.png", # debug image of cc contours
    "thickness_image": "qc_snapshots/callosum_thickness.png", # whippersnappy 3D image of cc thickness
    "cc_html": "qc_snapshots/corpus_callosum.html", # plotly cc visualization
    ## surface
    "surf_file": "surf/callosum.surf", # cc surface file
    "overlay_file": "surf/callosum.thickness.w", # cc surface overlay file
    "vtk_file": "surf/callosum_mesh.vtk", # vtk file of cc mesh
}