#!/bin/bash
#SBATCH --nodes 1          # Request 1 nodes so all resources are in two nodes.
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-02:50
#SBATCH --output=%N-%j.out
#SBATCH --account=def-jbpoline

# cd "$( dirname "${BASH_SOURCE[0]}" )"

module load singularity/3.6

singularity exec --nv --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1_ses2_0_bids.squashfs:ro ../../FastSurfer.sif ./run_inference_ukb_gpu.sh
