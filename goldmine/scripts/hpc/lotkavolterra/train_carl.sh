#!/bin/bash

#SBATCH --job-name=train_carl
#SBATCH --output=log_train_carl_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

#./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000 --trainsample train_zoom --modellabel model_zoom --hidden 3
#./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000 --trainsample train_zoom --modellabel model_zoom --hidden 3
#./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000 --trainsample train_zoom --modellabel model_zoom --hidden 3
#./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000 --trainsample train_zoom --modellabel model_zoom --hidden 3
#./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000 --trainsample train_zoom --modellabel model_zoom --hidden 3
#./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000 --trainsample train_zoom --modellabel model_zoom --hidden 3
#./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 100000 --trainsample train_zoom --modellabel model_zoom --hidden 3
#./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 200000 --trainsample train_zoom --modellabel model_zoom --hidden 3
./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 500000 --trainsample train_zoom --modellabel model_zoom --hidden 3
./train.py lotkavolterra carl -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000000 --trainsample train_zoom --modellabel model_zoom --hidden 3
