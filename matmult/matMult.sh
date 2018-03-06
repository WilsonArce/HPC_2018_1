#!/bin/bash
#
#SBATCH --job-name=matMult
#SBATCH --output=res_matMult.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./matMult $1 $2 $3 $4