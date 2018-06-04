#!/bin/bash

#SBATCH --job-name=DisplayImage
#SBATCH --output=outDisplayImage.txt
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

# cmake .
# make
./DisplayImage secret2.jpg cover2.jpg