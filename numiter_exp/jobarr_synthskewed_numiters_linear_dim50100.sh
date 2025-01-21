#!/bin/bash
#SBATCH -J synthske-numiter-linear
#SBATCH --array=1-20
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=8:00:00
#SBATCH --output=./Sbatch-reps2/synthske-numiter-linear.%A\_%a.out
#SBATCH --error=./Sbatch-reps2/synthske-numiter-linear.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

n=$SLURM_ARRAY_TASK_ID
echo "parameters for iteration: ${iteration}"

data="synth-skewed"
prob="linear"
temp="1"
cc="config_seedproddim_dim100"
emb_seed="23"

python main_run_numiter.py --data ${data} --prob ${prob} --temp ${temp} --common_config ${cc} --runnum ${n} --emb_seed ${emb_seed}