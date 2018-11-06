#!/bin/bash

#SBATCH --job-name=sim-test1t
#SBATCH --output=log_simulate_test_singletheta_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=3-00:00:00

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./simulate.py --singletheta --nsamples 10000 --checkpoint --noscore --noratio lotkavolterra test_zoom_${SLURM_ARRAY_TASK_ID}
