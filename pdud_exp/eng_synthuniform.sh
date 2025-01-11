#!/bin/bash
#SBATCH -J suni-eng
#SBATCH --array=1-35
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00:00
#SBATCH --output=./Sbatch-eng/suni-eng-pdutil.%A\_%a.out
#SBATCH --error=./Sbatch-eng/suni-eng-pdutil.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

n=$SLURM_ARRAY_TASK_ID
iteration=`sed -n "${n} p" synthuniform_smlinrand.csv`      # Get n-th line (1-indexed) of the file
echo "parameters for iteration: ${iteration}"

data=$(echo ${iteration} | cut -d "," -f 1)
prob=$(echo ${iteration} | cut -d "," -f 2)
temp=$(echo ${iteration} | cut -d "," -f 3)
embseed=$(echo ${iteration} | cut -d "," -f 4)
cc="config_seedproddim_dim100"

python main_run_pdutils_seedwise.py --data ${data} --prob ${prob} --temp ${temp} --common_config ${cc} --emb_seed ${embseed}