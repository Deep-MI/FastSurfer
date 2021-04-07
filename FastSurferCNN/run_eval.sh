#!/bin/bash

python3 eval_with_tracker.py --i_dir /home/nikhil/projects/green_comp_neuro/FastSurfer/i_data/ \
	--o_dir ../data \
	--t sub-* \
	--in_name orig.nii.gz \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--tracker_log_dir './logs/track_sub_000_benchmark/' \
	--mock_run 0
