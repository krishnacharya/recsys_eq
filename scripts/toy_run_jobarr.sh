#SBATCH -toyrun_singlecore
#SBATCH -A gts-jziani3
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=krish.epurchase@gmail.com
#SBATCH --mem-per-cpu=1G
#SBATCH -t15


python main_run.py --data synth-uniform --prob softmax --temp 10 --common_config toy_config