#!/bin/bash
#SBATCH --job-name=shape_agi_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Load modules if required (depends on bw setup)
module purge
module load python/3.10

# Activate virtual environment
source ~/shape_agi_project/.venv/bin/activate

# Move to project directory
cd ~/shape_agi_project

# Create logs directory if not exists
mkdir -p logs

# Run training
python -m training.train \
  --config configs/base.yaml \
  --seed 42 \
  --output_dir results/cluster_run_${SLURM_JOB_ID}
