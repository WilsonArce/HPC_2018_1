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