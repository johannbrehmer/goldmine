#!/bin/bash

#SBATCH --job-name=sc1_train
#SBATCH --output=scandal_singletheta_train_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /home/jb6504/goldmine/goldmine

./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 100 --singletheta
./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 200 --singletheta
./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 500 --singletheta
./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000 --singletheta
./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000 --singletheta
./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000 --singletheta
./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000 --singletheta
./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000 --singletheta
./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000 --singletheta
./train.py epidemiology2d scandal -i ${SLURM_ARRAY_TASK_ID} --singletheta
