#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/epidemiology

sbatch --array=0-199 simulate_train.sh
sbatch --array=0-99 simulate_test_singletheta.sh
sbatch --array=0-99 simulate_test.sh
