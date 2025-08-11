#!/bin/bash
#SBATCH --job-name=phiid
#SBATCH --output=logs/phiid-%j.out
#SBATCH --error=logs/phiid-%j.err
#SBATCH --partition=h100-agentS-train
#SBATCH --gres=gpu:2

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate syn
export HF_ALLOW_CODE_EVAL=1

echo "Working directory: $(pwd)"
python /home/p84400019/projects/consciousness-llms/Synergy_Redundancy_LLMs/src/main_llm_execution_prompts.py