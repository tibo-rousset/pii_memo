#!/bin/bash
#SBATCH --job-name=injection_gen
#SBATCH --output=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=01:00:00
#SBATCH --account=def-sreddy

#SBATCH --cpus-per-task=1

module load python/3.13.2
module load cuda
module load scipy-stack
module load arrow

source $HOME/pii_memo/bin/activate

echo "Job starting on $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

python3 prepare_injection_from_config.py \
    --config config/frequency_comparison_config.json \
    --filepath ../data/dataset/data/train-00000-of-00001.parquet \
    --output ../data/frequency_comparison.json