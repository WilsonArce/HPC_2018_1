#!/bin/bash

#SBATCH --job-name=imgTransform
#SBATCH --output=res_imgTransform.txt
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./sobel_Gmem images/7680x5022.jpg

# FILES=../images/*
# n=20

# for f in $FILES
# do
#   file=${f##*/}
#   echo -n ${file%.*}","
#   for ((i = 1; i <= n; i++));
#   do
#     ./sobel_Gmem $f
#     if [ $i -lt $n ]
#     then
#       echo -n ","
#     fi
#   done
#   echo " "
# done
