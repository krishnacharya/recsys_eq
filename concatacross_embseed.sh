#!/bin/bash
#SBATCH -J concat_seeds
#SBATCH --array=1-5
#SBATCH -A gts-jziani3
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=./concat_seeds.%A_%a.out
#SBATCH --error=./concat_seeds.%A_%a.error

# Load the necessary module
module load anaconda3/2022.05.0.1
conda activate recsys

# Define the list of base directories
BASE_DIRS=(
    "saved_frames_engtorch_k1"
    "saved_frames_engtorch_k5"
    "saved_frames_engtorch_k10"
    "saved_frames_engtorch_k20"
    "saved_frames_engtorch_k100"
)

# Get the directory for this task array index
BASE_DIR=${BASE_DIRS[$SLURM_ARRAY_TASK_ID-1]}

# Call the Python script
echo "Processing directory: $BASE_DIR"
python concat_seeds.py --base_dir "$BASE_DIR"