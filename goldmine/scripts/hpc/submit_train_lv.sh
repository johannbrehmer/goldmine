#!/bin/bash

cd /home/jb6504/goldmine/goldmine/scripts/hpc

sbatch --array=0-4 train_lv_maf.sh
sbatch --array=0-4 train_lv_scandal.sh
