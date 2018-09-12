#!/bin/bash

#SBATCH --job-name=sim-train
#SBATCH --output=simulate_lv_train_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=5-00:00:00

source activate goldmine
cd /home/jb6504/goldmine/goldmine/

./simulate.py --nthetas 100 --nsamples 100 --noratio lotkavolterra train${SLURM_ARRAY_TASK_ID}
