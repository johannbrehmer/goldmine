#!/bin/bash

#SBATCH --job-name=train_chkpt
#SBATCH --output=train_checkpoint_scandalmog_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

./train.py lotkavolterra scandal --checkpoint -i ${SLURM_ARRAY_TASK_ID} --samplesize 100000 --alpha 0.01 --trainsample train_zoom --modellabel model_zoom_mog --components 10
