#!/bin/bash
#SBATCH -J rtr_numiSM
#SBATCH --array=1-40
#SBATCH --mem-per-cpu=8G
#SBATCH -A gts-jziani3
#SBATCH --time=2-00:00:00
#SBATCH --output=./Sbatch-bigdat/rtr_numiSM.%A\_%a.out
#SBATCH --error=./Sbatch-bigdat/rtr_numiSM.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

n=$SLURM_ARRAY_TASK_ID
echo "parameters for iteration: ${iteration}"

data="rentrunway"
prob="softmax"
temp="1"
cc="config_seedproddim_dim100"
emb_seed="17"

python main_run_numiter.py --data ${data} --prob ${prob} --temp ${temp} --common_config ${cc} --runnum ${n} --emb_seed ${emb_seed}
