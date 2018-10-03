#!/bin/bash

#SBATCH --job-name=test_groundtruth
#SBATCH --output=log_test_groundtruth_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
# #SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

./test.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --densitygrid --ratiogrid --density --score --classifiertest --testsample test_zoom --model model_zoom_groundtruth
