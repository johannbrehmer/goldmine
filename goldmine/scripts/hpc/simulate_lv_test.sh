#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=simulate_lv_test_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=5-00:00:00

source activate goldmine
cd /home/jb6504/goldmine/goldmine/

./simulate.py --singletheta --nsamples 100 --noratio lotkavolterra test${SLURM_ARRAY_TASK_ID}
