#!/bin/bash

#SBATCH --job-name=sim-train
#SBATCH --output=log_simulate_train_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=1-00:00:00

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./simulate.py --nthetas 2000 --nsamples 1 --checkpoint lotkavolterra train_zoom_${SLURM_ARRAY_TASK_ID}
