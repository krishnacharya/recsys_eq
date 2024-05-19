#!/bin/bash
#SBATCH -J synthuconcat
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=./Sbatch-reports/mlsynthconcatembseed.%A\_%a.out
#SBATCH --error=./Sbatch-reports/mlsynthconcatembseed.%A\_%a.error

module load anaconda3/2022.05.0.1
conda activate recsys

data="movielens-100k"

python main_concat_embseeds.py --data ${data}
