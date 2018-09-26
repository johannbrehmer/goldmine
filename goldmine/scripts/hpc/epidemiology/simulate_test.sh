#!/bin/bash

#SBATCH --job-name=epsimtest
#SBATCH --output=simulate_lv_test_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=2-00:00:00

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./simulate.py --nthetas 1000 --nsamples 1 --noscore --noratio epidemiology2d test${SLURM_ARRAY_TASK_ID}
