#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=0-9 test_lv_maf.sh
sbatch --array=0-5 test_lv_scandal.sh
