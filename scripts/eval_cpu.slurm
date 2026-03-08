#!/bin/bash
#SBATCH --job-name=shape_agi_eval
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Starting evaluation (CPU) on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load devel/python/3.10

cd ~/shape_agi_project

mkdir -p logs

source venv/bin/activate

# Force CPU - no GPU
export CUDA_VISIBLE_DEVICES=""

echo "Running evaluation (CPU mode)..."

DELAY_MS=$1
MODEL_PATH=$2
SEED=$3
NOISY=${4:-""}

NOISY_FLAG=""
if [ "$NOISY" = "noisy" ]; then
  NOISY_FLAG="--noisy"
fi

python -m evaluation.evaluate \
  --config configs/base.yaml \
  --model_path ${MODEL_PATH} \
  --delay_ms ${DELAY_MS} \
  --episodes 200 \
  --steps_per_episode 20 \
  --seed ${SEED} \
  ${NOISY_FLAG} \
  --output_dir results/eval_delay${DELAY_MS}_seed${SEED}_${SLURM_JOB_ID}

echo "Evaluation finished."
