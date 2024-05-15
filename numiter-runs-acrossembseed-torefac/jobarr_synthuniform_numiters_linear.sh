#!/bin/bash
#SBATCH -J synthuni-numiter-linear
#SBATCH --array=1-10
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=./Sbatch-reports/synthuni-numiter-linear.%A\_%a.out
#SBATCH --error=./Sbatch-reports/synthuni-numiter-linear.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

n=$SLURM_ARRAY_TASK_ID
echo "parameters for iteration: ${iteration}"

data="synth-uniform"
prob="linear"
temp="1"
cc="config_less_seeds"
emb_seed="29"

python main_run_numiter.py --data ${data} --prob ${prob} --temp ${temp} --common_config ${cc} --runnum ${n} --emb_seed ${emb_seed}
