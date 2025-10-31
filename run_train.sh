#!/bin/bash
#SBATCH --job-name=pythia_download
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --account=def-sreddy
#SBATCH --gpus:1

#SBATCH --mail-user=thibault.rousset@mail.mcgill.ca
#SBATCH --mail-type=END

mkdir -p logs

module load python/3.13.2
module load cuda
module load scipy-stack

source $HOME/pii_memo/bin/activate

echo "Job starting on $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Using GPU(s): $SLURM_GPUS_ON_NODE"

python3 simple_train.py --pile_data_path "$HOME/scratch/pii_memo/data/indicies.npy"