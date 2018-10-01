#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=0-4 train_maf.sh
sbatch --array=0-4 train_scandal.sh
sbatch --array=0-4 train_rascandal.sh

sbatch --array=0-4 train_rascal.sh
sbatch --array=0-4 train_cascal.sh
sbatch --array=0-4 train_rolr.sh
sbatch --array=0-4 train_carl.sh

sbatch --array=0-4 train_maf_singletheta.sh
sbatch --array=0-4 train_scandal_largealpha.sh
