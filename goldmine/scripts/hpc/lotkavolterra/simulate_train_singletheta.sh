#!/bin/bash

#SBATCH --job-name=sim-train1t
#SBATCH --output=log_simulate_train_singletheta_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=2-00:00:00

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./simulate.py --nsamples 2000 lotkavolterra train_zoom_${SLURM_ARRAY_TASK_ID} --singletheta
