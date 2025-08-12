#!/bin/bash

# Manually hardcoded values (safe to edit before submission)
MODEL="L32-1"                               # L32-1, D2-16-A2, P-1
GENERATION="original_prompts"              # subset, original_prompts
TIME_SERIES="attention_outputs"            # attention_outputs, expert_output
PHYID="base"                               # discrete
DEACTIVATION_ANALYSIS="base"         # reverse_kl, noisy

# Which scripts to run (set to true or false)
RUN_RECORD_ACTIVATIONS=false
RUN_TIME_SERIES=false
RUN_RANKED_DEACTIVATIONS=true

# GPU Partition 
PARTITION="agentS-long"  # cpu, agentS-long, agentS-xlong

# Generate a unique job script at submission time
TIMESTAMP=$(date +%s)
JOB_NAME="${MODEL}_${GENERATION}_${TIME_SERIES}_${PHYID}"
JOB_SCRIPT="scripts/run_${JOB_NAME}.sh"
LOG_FILE="logs/%j-${JOB_NAME}.out"

mkdir -p scripts logs

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=Φ-${JOB_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:h200:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate int
export HF_ALLOW_CODE_EVAL=1

echo "Working directory: \$(pwd)"
echo "Running with config: model=$MODEL generation=$GENERATION time_series=$TIME_SERIES phyid=$PHYID deactivation_analysis=$DEACTIVATION_ANALYSIS"

# Conditionally run each step
if $RUN_RECORD_ACTIVATIONS; then
  echo "▶ Running record_activations.py"
  python /home/p84400019/projects/consciousness-llms/IT-LLMs/scripts/record_activations.py model=$MODEL generation=$GENERATION time_series=$TIME_SERIES phyid=$PHYID deactivation_analysis=$DEACTIVATION_ANALYSIS
fi

if $RUN_TIME_SERIES; then
  echo "▶ Running time_series_and_phyid.py"
  python /home/p84400019/projects/consciousness-llms/IT-LLMs/scripts/time_series_and_phyid.py model=$MODEL generation=$GENERATION time_series=$TIME_SERIES phyid=$PHYID deactivation_analysis=$DEACTIVATION_ANALYSIS
fi

if $RUN_RANKED_DEACTIVATIONS; then
  echo "▶ Running ranked_deactivations.py"
  python /home/p84400019/projects/consciousness-llms/IT-LLMs/scripts/ranked_deactivations.py model=$MODEL generation=$GENERATION time_series=$TIME_SERIES phyid=$PHYID deactivation_analysis=$DEACTIVATION_ANALYSIS
fi
EOF

chmod +x "$JOB_SCRIPT"
sbatch "$JOB_SCRIPT"
rm "$JOB_SCRIPT"
