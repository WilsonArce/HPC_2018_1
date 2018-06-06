#!/bin/bash

#SBATCH --job-name=DisplayImage
#SBATCH --output=outDisplayImage.txt
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

# ./DisplayImage testImages/512x512.1.jpg testImages/512x512.jpg
# ./DisplayImage testImages/1200x800.1.jpg testImages/1200x800.jpg
# ./DisplayImage testImages/1600x784.1.jpg testImages/1600x784.jpg
# ./DisplayImage testImages/7680x5022.1.jpg testImages/7680x5022.jpg
# ./DisplayImage testImages/8000x4500.1.jpg testImages/8000x4500.jpg
# ./DisplayImage testImages/14080x7920.1.jpg testImages/14080x7920.jpg

n=10

#./DisplayImage images/secret2.jpg images/cover2.jpg
for ((i = 1; i <= n; i++));
do
  ./DisplayImage testImages/1200x800.1.jpg testImages/1200x800.jpg
  if [ $i -lt $n ]
  then
    echo " "
  fi
done

