#!/bin/bash
#SBATCH --job-name=Φ-plot
#SBATCH --output=logs/%j-plot.out
#SBATCH --error=logs/%j-plot_synergistic_core.out
#SBATCH --partition=agentS-xlong
#SBATCH --gres=gpu:h200:1
# h100-agentS-train or cpu

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate int
export HF_ALLOW_CODE_EVAL=1

echo "Working directory: $(pwd)"

python /home/p84400019/projects/consciousness-llms/IT-LLMs/scripts/plot_synergistic_core.py
