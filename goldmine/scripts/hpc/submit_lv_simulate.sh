#!/bin/bash

cd /home/jb6504/goldmine/goldmine/scripts/hpc

sbatch --array=0-99 simulate_lv_train.sh
sbatch --array=0-99 simulate_lv_test.sh
