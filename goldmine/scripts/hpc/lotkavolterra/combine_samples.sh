#!/bin/bash

#SBATCH --job-name=combine
#SBATCH --output=log_combine_samples.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=62GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./combine_samples.py --regex lotkavolterra train_zoom "train_zoom_\d+"
./combine_samples.py --regex lotkavolterra test_zoom_singletheta "test_zoom_\d+_singletheta"
./combine_samples.py --regex lotkavolterra test_zoom "test_zoom_\d+"
