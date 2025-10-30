#!/bin/bash
#SBATCH --job-name=pytorch-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=3G
#SBATCH --time=00:01:00
#SBATCH --account=def-sreddy

#SBATCH --mail-user=thibault.rousset@mail.mcgill.ca
#SBATCH --mail-type=END

mkdir -p logs

module load python/3.13.2
module load cuda
module load scipy-stack

source $HOME/pii_memo_env/bin/activate

echo "Job starting on $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Using GPU(s): $SLURM_GPUS_ON_NODE"

python3 simple_train.py