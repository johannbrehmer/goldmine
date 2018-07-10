#!/bin/bash

cd /home/jb6504/goldmine/goldmine/scripts/hpc

sbatch --array=1-5 train_maf.sh
sbatch --array=1-5 train_scandal.sh
sbatch --array=1-5 train_maf_singletheta.sh
sbatch --array=1-5 train_scandal_singletheta.sh
