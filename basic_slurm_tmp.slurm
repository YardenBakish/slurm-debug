#! /bin/sh

#SBATCH --job-name=registers
#SBATCH --output=registers.out # redirect stdout
#SBATCH --error=registers.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=100000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


python basic_slurm_script.py