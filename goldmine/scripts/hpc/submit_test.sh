#!/bin/bash

cd /home/jb6504/goldmine/goldmine/scripts/hpc

#sbatch --array=1-9 test_maf.sh
sbatch --array=1-9 test_scandal.sh
sbatch --array=1-9 test_maf_singletheta.sh
