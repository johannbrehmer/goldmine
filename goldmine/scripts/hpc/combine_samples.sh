#!/bin/bash

#SBATCH --job-name=combine
#SBATCH --output=combine_samples.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB
#SBATCH --time=1-00:00:00

source activate goldmine
cd /home/jb6504/goldmine/goldmine/

./combine_samples.py --regex lotkavolterra train_combined train\d+
