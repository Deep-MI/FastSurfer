#!/bin/bash

python3 eval.py --i_dir /neurohub/ukbb/imaging/sub-2017717/ses-2/anat/ \
	--o_dir ../data \
	--t '*' \
	--in_name sub-2017717_ses-2_T1w.nii.gz \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--prune_type layerwise \
	--prune_percent 0.1

