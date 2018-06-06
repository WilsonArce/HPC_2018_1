#!/bin/bash

#SBATCH --job-name=DisplayImage
#SBATCH --output=outDisplayImage.txt
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./DisplayImage images/secret2.jpg images/cover2.jpg

#./DisplayImage testImages/512x512.1.jpg testImages/512x512.jpg

#NOTA: implementar el siguiente codigo para pruebas y obtener tendencias-graficas
# n=10

# for ((i = 1; i <= n; i++));
# do
#   ./DisplayImage testImages/512x512.1.jpg testImages/512x512.jpg
#   if [ $i -lt $n ]
#   then
#     echo -n ","
#   fi
# done

