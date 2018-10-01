#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=0-4 test_maf_singletheta.sh
# sbatch --array=0-4 test_maf.sh
# sbatch --array=0-4 test_scandal.sh
# sbatch --array=0-4 test_rascandal.sh
# sbatch --array=0-4 test_scandalcv.sh

# sbatch --array=0-4 test_carl.sh
# sbatch --array=0-4 test_rolr.sh
# sbatch --array=0-4 test_cascal.sh
# sbatch --array=0-4 test_rascal.sh
