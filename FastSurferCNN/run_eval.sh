#!/bin/bash

python3 eval_with_tracker.py --i_dir /home/nikhil/projects/Parkinsons/data/fs60/NC_fmriprep_anat_20.2.0/freesurfer-6.0.1/ \
	--o_dir ../data \
	--t sub-0128 \
	--in_name mri/orig.mgz \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--tracker_log_dir './logs/tmp/' \
	--mock_run 4
