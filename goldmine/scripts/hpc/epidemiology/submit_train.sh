#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/epidemiology

# sbatch --array=0-4 train_maf.sh
sbatch --array=0-4 train_scandal.sh
# sbatch --array=0-0 train_rascandal.sh
