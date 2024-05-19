#!/bin/bash
#SBATCH -J embconcat
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=./Sbatch-reports/mlembconc.%A\_%a.out
#SBATCH --error=./Sbatch-reports/mlembconc.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

data="synth-uniform"

python main_concat_embseed.py --data ${data}