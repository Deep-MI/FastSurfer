#!/bin/bash
#SBATCH --nodes 1          # Request 1 nodes so all resources are in two nodes.
#SBATCH --gres=gpu       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-020:50
#SBATCH --output=%N-%j.out

module load python/3.7
source /home/nikhil/projects/def-jbpoline/nikhil/deep_learning/code/env/bin/activate

python3 eval.py --i_dir /home/nikhil/projects/Parkinsons/data/fs60/NC_fmriprep_anat_20.2.0/freesurfer-6.0.1/ \
	--o_dir ../data \
	--t sub-* \
	--in_name mri/orig.mgz \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl 