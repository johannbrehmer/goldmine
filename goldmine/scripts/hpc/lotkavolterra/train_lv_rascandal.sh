#!/bin/bash

#SBATCH --job-name=train_rascandal
#SBATCH --output=train_rascandal_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

./train.py lotkavolterra rascandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000 --alpha 0.01 --beta 0.01 --trainsample train_focus
./train.py lotkavolterra rascandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000 --alpha 0.01 --beta 0.01 --trainsample train_focus
./train.py lotkavolterra rascandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000 --alpha 0.01 --beta 0.01 --trainsample train_focus
./train.py lotkavolterra rascandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000 --alpha 0.01 --beta 0.01 --trainsample train_focus
./train.py lotkavolterra rascandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000 --alpha 0.01 --beta 0.01 --trainsample train_focus
./train.py lotkavolterra rascandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000 --alpha 0.01 --beta 0.01 --trainsample train_focus
./train.py lotkavolterra rascandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 100000 --alpha 0.01 --beta 0.01 --trainsample train_focus
./train.py lotkavolterra rascandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 200000 --alpha 0.01 --beta 0.01 --trainsample train_focus
