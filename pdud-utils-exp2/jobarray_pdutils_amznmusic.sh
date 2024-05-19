#!/bin/bash
#SBATCH -J amznmusic_pdutils
#SBATCH --array=1-11
#SBATCH --mem-per-cpu=10G
#SBATCH --time=2-00:00:00
#SBATCH --output=./Sbatch-reports/amznmusic_pdutils.%A\_%a.out
#SBATCH --error=./Sbatch-reports/amznmusic_pdutils.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

n=$SLURM_ARRAY_TASK_ID
iteration=`sed -n "${n} p" amznmusic_allseeds_smandlin.csv`      # Get n-th line (1-indexed) of the file
echo "parameters for iteration: ${iteration}"

data=$(echo ${iteration} | cut -d "," -f 1)
prob=$(echo ${iteration} | cut -d "," -f 2)
temp=$(echo ${iteration} | cut -d "," -f 3)
embseed=$(echo ${iteration} | cut -d "," -f 4)
cc="config_seedproddim"

python main_run_pdutils_seedwise.py --data ${data} --prob ${prob} --temp ${temp} --common_config ${cc} --emb_seed ${embseed}