#!/bin/bash 
#SBATCH -J toyslurm
#SBATCH -A gts-jziani3
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH -t15
#SBATCH --output=./Sbatch-reports/Array_test.%A.out
#SBATCH --error=./Sbatch-reports/Array_test.%A.error
module load anaconda3/2022.05.0.1
conda activate recsys
python main_run.py --data movielens-100k --prob softmax --temp 1