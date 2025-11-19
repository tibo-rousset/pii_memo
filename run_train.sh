#!/bin/bash
#SBATCH --job-name=pythia_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --account=def-sreddy
#SBATCH --gpus=1

#SBATCH --cpus-per-task=1

#SBATCH --mail-user=thibault.rousset@mail.mcgill.ca
#SBATCH --mail-type=END

mkdir -p logs

module load python/3.13.2
module load cuda
module load scipy-stack

source $HOME/pii_memo/bin/activate

echo "Job starting on $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Using GPU(s): $SLURM_GPUS_ON_NODE"

python3 -u src/simple_train.py --pile_data_path "$HOME/scratch/pii_memo/data/indicies.npy" \
    --config "$HOME/scratch/pii_memo/train_config.json"
    --no_download \
    --injection_data_path debug_injection.json \
    --inject_sequence_ids pii_sequences \
    --debug
