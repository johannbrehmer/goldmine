#!/bin/bash

cd /home/jb6504/goldmine/goldmine/scripts/hpc

sbatch --array=1-3 train_lv_maf.sh
sbatch --array=1-3 train_lv_scandal.sh
