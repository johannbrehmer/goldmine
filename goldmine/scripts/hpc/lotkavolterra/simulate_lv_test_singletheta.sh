#!/bin/bash

#SBATCH --job-name=sim-test1t
#SBATCH --output=simulate_lv_test_singletheta_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=2-00:00:00

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./simulate.py --singletheta --nsamples 1000 --noscore --noratio lotkavolterra testfocus${SLURM_ARRAY_TASK_ID}
