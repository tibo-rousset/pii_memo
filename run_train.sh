#!/bin/bash
#SBATCH --job-name=pii-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:01:00

mkdir -p logs

module load python/3.13.2
module load cuda
module load scipy-stack

source $HOME/pii_memo_env/bin/activate

echo "Job starting on $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Using GPU(s): $SLURM_GPUS_ON_NODE"

