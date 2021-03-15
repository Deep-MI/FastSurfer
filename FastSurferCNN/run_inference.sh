#!/bin/bash
#SBATCH --nodes 1          # Request 1 nodes so all resources are in two nodes.
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-02:50
#SBATCH --output=%N-%j.out

module load python/3.7
source /home/nikhil/projects/def-jbpoline/nikhil/deep_learning/code/env/bin/activate

echo prune_percent 0.75
python3 eval.py --i_dir /home/nikhil/projects/def-jbpoline/nikhil/Parkinsons/data/freesurfer/PD_fmriprep_anat_20.2.0/freesurfer-6.0.1/ \
	--o_dir ../data/prune_75/PD/ \
	--t sub-* \
	--in_name mri/orig.mgz \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--prune_type layerwise \
	--prune_percent 0.75

echo prune_percent 0.5
python3 eval.py --i_dir /home/nikhil/projects/def-jbpoline/nikhil/Parkinsons/data/freesurfer/PD_fmriprep_anat_20.2.0/freesurfer-6.0.1/ \
	--o_dir ../data/prune_50/PD/ \
	--t sub-* \
	--in_name mri/orig.mgz \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--prune_type layerwise \
	--prune_percent 0.5

echo prune_percent 0.25
python3 eval.py --i_dir /home/nikhil/projects/def-jbpoline/nikhil/Parkinsons/data/freesurfer/PD_fmriprep_anat_20.2.0/freesurfer-6.0.1/ \
	--o_dir ../data/prune_25/PD/ \
	--t sub-* \
	--in_name mri/orig.mgz \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--prune_type layerwise \
	--prune_percent 0.25

echo prune_percent 0.10
python3 eval.py --i_dir /home/nikhil/projects/def-jbpoline/nikhil/Parkinsons/data/freesurfer/PD_fmriprep_anat_20.2.0/freesurfer-6.0.1/ \
	--o_dir ../data/prune_10/PD/ \
	--t sub-* \
	--in_name mri/orig.mgz \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--prune_type layerwise \
	--prune_percent 0.10