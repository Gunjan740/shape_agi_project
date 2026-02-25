#!/bin/bash
#SBATCH --job-name=shape_agi_eval
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Starting evaluation on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"

module purge
module load devel/cuda/11.8
module load devel/python/3.10

cd ~/shape_agi_project

mkdir -p logs

source venv/bin/activate

nvidia-smi

echo "Running evaluation..."

DELAY_MS=$1
MODEL_PATH=$2
SEED=$3

python -m evaluation.evaluate \
  --config configs/base.yaml \
  --model_path ${MODEL_PATH} \
  --delay_ms ${DELAY_MS} \
  --episodes 50 \
  --steps_per_episode 30 \
  --seed ${SEED} \
  --output_dir results/eval_delay_${DELAY_MS}_seed_${SEED}_${SLURM_JOB_ID}

echo "Evaluation finished."
