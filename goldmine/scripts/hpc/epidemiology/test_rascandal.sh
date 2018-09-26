#!/bin/bash

#SBATCH --job-name=scandal_eval
#SBATCH --output=scandal_test_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 100 --classifiertest
./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 200 --classifiertest
./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 500 --classifiertest
./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000 --classifiertest
./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000 --classifiertest
./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000 --classifiertest
./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000 --classifiertest
./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000 --classifiertest
./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000 --classifiertest
./test.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --classifiertest
