#!/bin/bash
#SBATCH -J jobarray_demo
#SBATCH --array=1-40
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=./Sbatch-rep/Array_test.%A\_%a.out
#SBATCH --error=./Sbatch-rep/Array_test.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

n=$SLURM_ARRAY_TASK_ID
iteration=`sed -n "${n} p" dimseed_full.csv`      # Get n-th line (1-indexed) of the file
echo "parameters for iteration: ${iteration}"

dim=$(echo ${iteration} | cut -d "," -f 1)
seed=$(echo ${iteration} | cut -d "," -f 2)

python maingen_rentrunway.py --dimension ${dim} --seed ${seed}