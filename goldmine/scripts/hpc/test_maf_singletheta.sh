#!/bin/bash

#SBATCH --job-name=maf1_eval
#SBATCH --output=maf1_test_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00

source activate goldmine
cd /home/jb6504/goldmine/goldmine

./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 100 --classifiertest --singletheta
./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 200 --classifiertest --singletheta
./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 500 --classifiertest --singletheta
./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000 --classifiertest --singletheta
./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000 --classifiertest --singletheta
./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000 --classifiertest --singletheta
./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000 --classifiertest --singletheta
./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000 --classifiertest --singletheta
./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000 --classifiertest --singletheta
./test.py epidemiology2d maf -i ${SLURM_ARRAY_TASK_ID} --classifiertest --singletheta
