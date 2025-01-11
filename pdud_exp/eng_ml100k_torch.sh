#!/bin/bash
#SBATCH -J ml100ktopk
#SBATCH --array=1-100
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --output=./Sbatch-engtorch/ml100ktopk-pdutil.%A\_%a.out
#SBATCH --error=./Sbatch-engtorch/ml100ktopk-pdutil.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

n=$SLURM_ARRAY_TASK_ID
iteration=`sed -n "${n} p" ml100k_testtopk.csv`      # Get n-th line (1-indexed) of the file
echo "parameters for iteration: ${iteration}"

data=$(echo ${iteration} | cut -d "," -f 1)
prob=$(echo ${iteration} | cut -d "," -f 2)
temp=$(echo ${iteration} | cut -d "," -f 3)
topk=$(echo ${iteration} | cut -d "," -f 4)
embseed=$(echo ${iteration} | cut -d "," -f 5)
cc="config_seedproddim_dim100"
save_dir="../saved_frames_engtorch_k${topk}/"

python main_run_pdutils_seedwise.py --save_dir ${save_dir} --data ${data} --prob ${prob} --temp ${temp} --common_config ${cc} --emb_seed ${embseed} --topk ${topk}