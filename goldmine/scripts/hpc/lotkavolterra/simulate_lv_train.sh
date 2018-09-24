#!/bin/bash

#SBATCH --job-name=sim-train
#SBATCH --output=simulate_lv_train_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=2-00:00:00

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./simulate.py --nthetas 2200 --nsamples 1 lotkavolterra train${SLURM_ARRAY_TASK_ID}
