#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=0-9 train_checkpoint_scandalmog.sh

# sbatch --array=0-9 train_mafmog.sh
# sbatch --array=0-9 train_scandalmog.sh
# sbatch --array=0-9 train_scandalmog_largealpha.sh
# sbatch --array=0-9 train_scandalmog_smallalpha.sh

# sbatch --array=0-9 train_rascal.sh
# sbatch --array=0-9 train_cascal.sh
# sbatch --array=0-9 train_rolr.sh
# sbatch --array=0-9 train_carl.sh
