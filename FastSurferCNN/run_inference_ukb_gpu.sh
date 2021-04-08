#!/bin/bash

DATA_DIR='/neurohub/ukbb/imaging/'
INPUT_FILE_NAME='ses-2/anat/sub-2017717_ses-2_T1w.nii.gz'
PARAM_T='sub-2017717'

# cd "$( dirname "${BASH_SOURCE[0]}" )"

python3 eval_with_tracker.py --i_dir "${DATA_DIR}" \
	--o_dir ../data/ukb/ \
	--t "${PARAM_T}" \
	--in_name "${INPUT_FILE_NAME}" \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--geo_loc '45.4972159,-73.6103642' \
	--prune_type layerwise \
	--prune_percent 0.0 \
	--tracker_log_dir './logs/CC_ukb_gpu_test/' \
	--mock_run 0
