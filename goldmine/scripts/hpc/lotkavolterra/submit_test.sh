#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=0-9 test_mafmog.sh
sbatch --array=0-9 test_scandalmog.sh
sbatch --array=0-9 test_scandalmog_largealpha.sh
sbatch --array=0-9 test_scandalmog_smallalpha.sh

sbatch --array=0-9 test_rascal.sh
sbatch --array=0-9 test_cascal.sh
sbatch --array=0-9 test_rolr.sh
sbatch --array=0-9 test_carl.sh