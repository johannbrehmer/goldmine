#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=250-499 simulate_train.sh
# sbatch --array=0-49 simulate_train_singletheta.sh
# sbatch --array=0-9 simulate_test.sh
# sbatch --array=0-9 simulate_test_singletheta.sh
