#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=simulate_lv_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=4-00:00:00
# #SBATCH --gres=gpu:1

cd /home/jb6504/goldmine/goldmine/

./simulate.py --nthetas 10000 --nsamples 100 --noratio lotkavolterra train
./simulate.py --singletheta --nsamples 100000 --noratio lotkavolterra test
