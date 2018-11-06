#!/bin/bash

#SBATCH --job-name=test_cascal
#SBATCH --output=log_test_cascal_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
# #SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

#./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000 --ratiogrid  --testsample test_zoom --model model_zoom
#./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000 --ratiogrid  --testsample test_zoom --model model_zoom
#./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000 --ratiogrid  --testsample test_zoom --model model_zoom
#./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000 --ratiogrid  --testsample test_zoom --model model_zoom
#./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000 --ratiogrid  --testsample test_zoom --model model_zoom
#./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000 --ratiogrid  --testsample test_zoom --model model_zoom
#./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 100000 --ratiogrid  --testsample test_zoom --model model_zoom
#./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 200000 --ratiogrid  --testsample test_zoom --model model_zoom
./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 500000 --ratiogrid  --testsample test_zoom --model model_zoom
#./test.py lotkavolterra cascal -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000000 --ratiogrid  --testsample test_zoom --model model_zoom
