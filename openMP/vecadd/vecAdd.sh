#!/bin/bash
#
#SBATCH --job-name=vecAdd
#SBATCH --output=res_vecAdd.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./vecAdd $1
