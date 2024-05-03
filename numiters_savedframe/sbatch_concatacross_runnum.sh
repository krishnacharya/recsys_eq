#!/bin/bash
#SBATCH -J ml-runconcat
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=./Sbatch-reports/ml-runconcat.%A\_%a.out
#SBATCH --error=./Sbatch-reports/ml-runconcat.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

data="movielens-100k"

python main_concat_runnum.py --data ${data}