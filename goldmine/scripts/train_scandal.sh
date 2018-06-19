#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology scandal --samplesize 100 --epochs 10 --compensate_samplesize
./train.py epidemiology scandal --samplesize 200 --epochs 10 --compensate_samplesize
./train.py epidemiology scandal --samplesize 500 --epochs 10 --compensate_samplesize
./train.py epidemiology scandal --samplesize 1000 --epochs 10 --compensate_samplesize
./train.py epidemiology scandal --samplesize 2000 --epochs 10 --compensate_samplesize
./train.py epidemiology scandal --samplesize 5000 --epochs 10 --compensate_samplesize
./train.py epidemiology scandal --samplesize 10000 --epochs 10 --compensate_samplesize
./train.py epidemiology scandal --samplesize 20000 --epochs 10 --compensate_samplesize
./train.py epidemiology scandal --samplesize 50000 --epochs 10 --compensate_samplesize
./train.py epidemiology scandal --epochs 10 --compensate_samplesize
