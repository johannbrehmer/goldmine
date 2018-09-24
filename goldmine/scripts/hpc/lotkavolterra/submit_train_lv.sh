#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

#sbatch --array=0-9 train_lv_maf.sh
#sbatch --array=0-9 train_lv_scandal.sh
sbatch --array=0-4 train_lv_scandal_largealpha.sh
sbatch --array=0-4 train_lv_scandal_smallalpha.sh
