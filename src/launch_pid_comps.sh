#!/bin/bash
#SBATCH --job-name=phiid
#SBATCH --output=logs/phiid-%j.out
#SBATCH --error=logs/phiid-%j.err
#SBATCH --partition=cpu
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=80

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate syn
export HF_ALLOW_CODE_EVAL=1

echo "Working directory: $(pwd)"

# List of models
FINAL_MODELS=('L32-1' 'L32-3' 'L31-8' 'L31-8-IT' 'L31-8-R1-Distill')

# Select model index: use SLURM_ARRAY_TASK_ID or custom ENV var MODEL_IDX
INDEX=1
MODEL_CODE="${FINAL_MODELS[$INDEX]}"
export MODEL_CODE 

echo "Running model at index $INDEX: $MODEL_CODE"
python /home/p84400019/projects/consciousness-llms/Synergy_Redundancy_LLMs/src/main_pid_computation_prompts.py
