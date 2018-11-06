#!/bin/bash

#SBATCH --job-name=sim-test
#SBATCH --output=log_simulate_test_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=3-00:00:00

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./simulate.py --nthetas 10000 --nsamples 1 --checkpoint --noscore --noratio lotkavolterra test_zoom_${SLURM_ARRAY_TASK_ID}
