#!/bin/bash

#SBATCH --job-name=train_maf
#SBATCH --output=train_maf_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /home/jb6504/goldmine/goldmine

./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID}
