#!/bin/bash
#SBATCH --job-name=shape_agi_reinforce
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"

module purge
module load devel/cuda/11.8
module load devel/python/3.10

cd ~/shape_agi_project

mkdir -p logs

source venv/bin/activate

nvidia-smi

echo "Running training (REINFORCE)..."

python -m training.train_reinforce \
  --config configs/base.yaml \
  --seed 123 \
  --output_dir results/cluster_run_${SLURM_JOB_ID}

echo "Job finished."
