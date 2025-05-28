#!/bin/bash
#SBATCH --job-name=phiid
#SBATCH --output=logs/phiid-%j.out
#SBATCH --error=logs/phiid-%j.err
#SBATCH --partition=cpu
#SBATCH --time=5-00:00:00   # 10 days

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate syn
export HF_ALLOW_CODE_EVAL=1

echo "Working directory: $(pwd)"

# Run training script with Hydra config overrides
MODEL_CODE="Q3-8" python /home/p84400019/projects/consciousness-llms/Synergy_Redundancy_LLMs/src/main_pid_computation_prompts.py