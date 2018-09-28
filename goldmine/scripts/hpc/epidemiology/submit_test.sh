#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/epidemiology

sbatch --array=0-4 test_maf.sh
sbatch --array=0-4 test_scandal.sh
# sbatch --array=0-4 test_rascandal.sh
