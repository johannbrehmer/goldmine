#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=0-99 simulate_train.sh
sbatch --array=0-9 simulate_test.sh
sbatch --array=0-9 simulate_test_singletheta.sh
