#!/bin/bash

cd /home/jb6504/goldmine/goldmine/scripts/hpc

sbatch --array=1-9 train_maf.sh
sbatch --array=1-9 train_scandal.sh
sbatch --array=1-9 train_maf_singletheta.sh
