#!/bin/bash

#SBATCH --job-name=test_scandal
#SBATCH --output=test_scandal_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000
./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000
./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000
./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000
./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000
./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000
./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID}
