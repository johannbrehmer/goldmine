#!/bin/bash

#SBATCH --job-name=test_scandalmog_largealpha
#SBATCH --output=log_test_scandalmog_largealpha_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
# #SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

#./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000 --densitygrid --ratiogrid --density --score  --testsample test_zoom --model model_zoom_mog_largealpha
#./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 2000 --densitygrid --ratiogrid --density --score --testsample test_zoom --model model_zoom_mog_largealpha
#./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 5000 --densitygrid --ratiogrid --density --score --testsample test_zoom --model model_zoom_mog_largealpha
#./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 10000 --densitygrid --ratiogrid --density --score --testsample test_zoom --model model_zoom_mog_largealpha
#./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 20000 --densitygrid --ratiogrid --density --score --testsample test_zoom --model model_zoom_mog_largealpha
#./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 50000 --densitygrid --ratiogrid --density --score --testsample test_zoom --model model_zoom_mog_largealpha
#./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 100000 --densitygrid --ratiogrid --density --score --testsample test_zoom --model model_zoom_mog_largealpha
#./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 200000 --densitygrid --ratiogrid --density --score --testsample test_zoom --model model_zoom_mog_largealpha
./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 500000 --densitygrid --ratiogrid --density --score --testsample test_zoom --model model_zoom_mog_largealpha
#./test.py lotkavolterra scandal -i ${SLURM_ARRAY_TASK_ID} --samplesize 1000000 --densitygrid --ratiogrid --density --score --testsample test_zoom --model model_zoom_mog_largealpha
