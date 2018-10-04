#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

# sbatch --array=0-0 train_groundtruth.sh

# sbatch --array=0-9 train_maf.sh
# sbatch --array=0-9 train_scandal.sh
# sbatch --array=0-9 train_rascandal.sh

# sbatch --array=0-9 train_mafmog.sh
sbatch --array=0-9 train_scandalmog.sh
# sbatch --array=0-9 train_rascandalmog.sh

# sbatch --array=0-9 train_rascal.sh
# sbatch --array=0-9 train_cascal.sh
# sbatch --array=0-9 train_rolr.sh
# sbatch --array=0-9 train_carl.sh

# sbatch --array=0-9 train_scandal_largealpha.sh
# sbatch --array=0-9 train_scandalmog_largealpha.sh
# sbatch --array=0-9 train_scandal_smallalpha.sh
# sbatch --array=0-9 train_scandalmog_smallalpha.sh
# sbatch --array=0-9 train_scandal_zeroalpha.sh
# sbatch --array=0-9 train_scandalmog_zeroalpha.sh

# sbatch --array=1-9 train_groundtruth.sh
