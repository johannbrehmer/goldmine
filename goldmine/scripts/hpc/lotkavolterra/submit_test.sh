#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=0-9 train_maf.sh
sbatch --array=0-9 train_scandal.sh
sbatch --array=0-9 train_rascandal.sh

sbatch --array=0-9 train_mafmog.sh
sbatch --array=0-9 train_scandalmog.sh
sbatch --array=0-9 train_rascandalmog.sh

sbatch --array=0-4 train_rascal.sh
sbatch --array=0-4 train_cascal.sh
sbatch --array=0-4 train_rolr.sh
sbatch --array=0-4 train_carl.sh

# sbatch --array=0-2 train_maf_singletheta.sh
# sbatch --array=0-2 train_mafmog_singletheta.sh
# sbatch --array=0-2 train_scandal_largealpha.sh
# sbatch --array=0-2 train_scandalmog_largealpha.sh
