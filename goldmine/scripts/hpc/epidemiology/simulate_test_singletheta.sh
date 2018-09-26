#!/bin/bash

#SBATCH --job-name=epsimt1t
#SBATCH --output=simulate_test_singletheta_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=2-00:00:00

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./simulate.py --singletheta --nsamples 1000 --noscore --noratio epidemiology2d test${SLURM_ARRAY_TASK_ID}
