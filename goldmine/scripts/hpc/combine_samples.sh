#!/bin/bash

#SBATCH --job-name=combine
#SBATCH --output=combine_samples.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=62GB
#SBATCH --time=1-00:00:00

source activate goldmine
cd /home/jb6504/goldmine/goldmine/

./combine_samples.py --regex lotkavolterra train "train\d+"
./combine_samples.py --regex lotkavolterra test_singletheta "test\d+_singletheta"
