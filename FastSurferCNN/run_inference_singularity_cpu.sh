#!/bin/bash
#SBATCH --nodes 1          # Request 1 nodes so all resources are in two nodes.
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-05:50
#SBATCH --output=%N-%j.out
#SBATCH --account=def-jbpoline
#SBATCH --array=0-72

# cd "$( dirname "${BASH_SOURCE[0]}" )"

[[ -z "${SLURM_ARRAY_TASK_ID}" ]] && exit 1

subj=$(awk -v row="${SLURM_ARRAY_TASK_ID}" \
    'BEGIN {FS=","} NR == row + 2 {print $5}' ~/fastsurfer_pilot_subjects.csv)
[[ -z "$subj" ]] && exit 1
echo "running inference for subject $subj"

module load singularity/3.6

singularity exec --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1_ses2_0_bids.squashfs:ro ../../FastSurfer.sif ./run_inference_ukb_cpu.sh
