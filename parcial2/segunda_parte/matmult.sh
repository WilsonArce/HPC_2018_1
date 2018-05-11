#!/bin/bash

#SBATCH --job-name=matmult
#SBATCH --output=out_matmult.txt
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

#./matmult $1
# file="ansTime.txt"
declare -a OUT
declare -a ANS

declare -a aux
size=(128 256 512 1024)
n=4

# data=($(./matmult 256))
# echo "${data[0]}"
for i in "${size[@]}"
do
  for ((j = 0; j < n; j++));
  do
    OUT=($(./matmult $i))
    for ((j = 1; j < 4; j++))
    do
      (($aux+=1))
      echo $aux
    done
  done
  ANS[0]=${OUT[0]}
  ANS[1]=${ANS[1]/$n}
  ANS[2]=${ANS[2]/$n}
  ANS[3]=${ANS[3]/$n}
  ANS[4]=${ANS[4]/$n}
  #echo ${ANS[@]}
  echo ""
done