#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=0-4 test_maf.sh
sbatch --array=0-4 test_scandal.sh
sbatch --array=0-4 test_rascandal.sh
# sbatch --array=0-4 test_scandalcv.sh
