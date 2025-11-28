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

from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT

### Constants
WEIGHTS_PATH = FASTSURFER_ROOT / "checkpoints"
FSAVERAGE_CENTROIDS_PATH = FASTSURFER_ROOT / "CorpusCallosum" / "data" / "fsaverage_centroids.json"
# Contains both affine and header
FSAVERAGE_DATA_PATH = FASTSURFER_ROOT / "CorpusCallosum" / "data" / "fsaverage_data.json"
FSAVERAGE_MIDDLE = 128  # Middle slice index in fsaverage space
CC_LABEL = 192          # Label value for corpus callosum in segmentation
FORNIX_LABEL = 250      # Label value for fornix in segmentation
THIRD_VENTRICLE_LABEL = 4 # Label value for third ventricle in segmentation
SUBSEGMENT_LABELS = [251, 252, 253, 254, 255] # labels for subsegments in segmentation


DEFAULT_INPUT_PATHS = {
    "conf_name": "mri/orig.mgz",
    "aseg_name": "mri/aparc.DKTatlas+aseg.deep.mgz",
}

DEFAULT_OUTPUT_PATHS = {
    ## images
    "upright_volume": None,  # orig.mgz mapped to upright space
    ## segmentations
    "segmentation": "mri/callosum.CC.upright.mgz",  # corpus callosum segmentation in upright space
    "segmentation_in_orig": "mri/callosum.CC.orig.mgz",  # cc segmentation in input segmentations space
    "softlabels_cc": "mri/callosum.CC.soft.mgz",  # cc softlabels  in upright space
    "softlabels_fn": "mri/fornix.CC.soft.mgz",  # fornix softlabels in upright space
    "softlabels_background": "mri/background.CC.soft.mgz",  # background softlabels in upright space
    ## stats
    "cc_markers": "stats/callosum.CC.midslice.json",  # cc metrics for middle slice
    "cc_measures": "stats/callosum.CC.all_slices.json",  # cc metrics for all slices
    ## transforms
    "upright_lta": "mri/transforms/cc_up.lta",  # lta transform from orig to upright space
    "orient_volume_lta": "mri/transforms/orient_volume.lta",  # lta transform from orig to upright+acpc corrected space
    ## qc
    "qc_image": None, #"callosum.png",  # debug image of cc contours
    "thickness_image": None, # "callosum.thickness.png",  # whippersnappy 3D image of cc thickness
    "cc_html": None, # "corpus_callosum.html",  # plotly cc visualization
    ## surface
    "cc_surf": "surf/callosum.surf",  # cc surface file
    "cc_thickness_overlay": "surf/callosum.thickness.w",  # cc surface overlay file
    "cc_surf_vtk": "surf/callosum.vtk",  # vtk file of cc mesh
}