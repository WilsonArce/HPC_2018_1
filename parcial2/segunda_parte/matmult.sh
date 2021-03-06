#!/bin/bash

#SBATCH --job-name=matmult
#SBATCH --output=out_matmult.txt
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./matmult $1

# declare -a OUT
# declare -a ANS

# aux=0
# size=(64 128 256 512 1024)
# n=10

rm "ansTime.txt"
# for i in "${size[@]}"
# do
#   for ((j = 0; j < n; j++));
#   do
#     ./matmult $i
#   done
# done