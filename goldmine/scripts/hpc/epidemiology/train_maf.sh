#!/bin/bash

#SBATCH --job-name=eptrainmaf
#SBATCH --output=train_maf_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 100
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 200
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 500
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 100000
./train.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 200000
