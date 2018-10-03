#!/bin/bash

#SBATCH --job-name=train_mafmog_singletheta
#SBATCH --output=train_mafmog_singletheta_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

./train.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --singletheta --trainsample train_zoom --modellabel model_zoom_mog --components 10
