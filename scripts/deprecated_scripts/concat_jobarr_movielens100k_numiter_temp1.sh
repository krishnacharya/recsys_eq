#!/bin/bash
#SBATCH -J movielens100k-concat
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=./Sbatch-reports/Array_test.%A\_%a.out
#SBATCH --error=./Sbatch-reports/Array_test.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

echo "parameters for iteration: ${iteration}"

data="movielens-100k"
prob="softmax"
temp="1"

python concat_runframes.py --data ${data} --prob ${prob} --temp ${temp}