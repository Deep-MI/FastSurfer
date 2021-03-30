#!/bin/bash

DATA_DIR='/neurohub/ukbb/imaging/'
INPUT_FILE_NAME='ses-2/anat/sub-2017717_ses-2_T1w.nii.gz'
PARAM_T='sub-2017717'

# cd "$( dirname "${BASH_SOURCE[0]}" )"

echo prune_percent 0.50 cohort NC
python3 eval.py --i_dir "${DATA_DIR}" \
	--o_dir ../data/prune_50/NC/ \
	--t "${PARAM_T}" \
	--in_name "${INPUT_FILE_NAME}" \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--prune_type layerwise \
	--prune_percent 0.5
