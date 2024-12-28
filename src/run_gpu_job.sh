#!/bin/bash
#SBATCH --job-name=my_gpu_job        # Job name
#SBATCH --output=output_%j.txt       # Standard output and error log (%j expands to jobID)
#SBATCH --mail-type=ALL              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pu22@ic.ac.uk  # Where to send mail
#SBATCH --ntasks=1                   # Run on a single CPU
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16gb                   # Job memory request
#SBATCH --time=02:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:teslaa40:1                 # Request 1 GPU
#SBATCH --partition=gpgpu            # Partition with big GPUs (e.g., Tesla A40)

# Load necessary modules and activate environment
source /vol/cuda/12.5.0/setup.sh     # Load CUDA toolkit
export PATH=/vol/bitbucket/pu22/myvenv/bin/:$PATH
source activate                      # Activate your virtual environment

# Navigate to the directory where you submitted the job
cd /vol/bitbucket/pu22/Synergy_Redundancy_LLMs/src/

# Run your Python script
# python3 gpu_test.py
python3 main_ablations.py

