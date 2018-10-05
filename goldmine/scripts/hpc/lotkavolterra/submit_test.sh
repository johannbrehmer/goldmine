#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

# sbatch --array=0-9 test_maf.sh
# sbatch --array=0-9 test_scandal.sh
# sbatch --array=0-9 test_rascandal.sh

# sbatch --array=0-9 test_mafmog.sh
sbatch --array=4-9 test_scandalmog.sh
# sbatch --array=0-9 test_rascandalmog.sh

# sbatch --array=0-9 test_rascal.sh
# sbatch --array=0-9 test_cascal.sh
# sbatch --array=0-9 test_rolr.sh
# sbatch --array=0-9 test_carl.sh

# sbatch --array=0-9 test_scandal_largealpha.sh
# sbatch --array=0-9 test_scandalmog_largealpha.sh
# sbatch --array=0-9 test_scandal_smallalpha.sh
# sbatch --array=0-9 test_scandalmog_smallalpha.sh
# sbatch --array=0-9 test_scandal_zeroalpha.sh
# sbatch --array=0-9 test_scandalmog_zeroalpha.sh

# sbatch --array=0-9 test_groundtruth.sh
