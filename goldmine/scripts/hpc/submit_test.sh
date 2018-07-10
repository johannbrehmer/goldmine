#!/bin/bash

cd /home/jb6504/goldmine/goldmine/scripts/hpc

sbatch --array=1-5 test_maf.sh
sbatch --array=1-5 test_scandal.sh
