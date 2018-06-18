#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology scandal --samplesize 100 --compensate_samplesize
./train.py epidemiology scandal --samplesize 200 --compensate_samplesize
./train.py epidemiology scandal --samplesize 500 --compensate_samplesize
./train.py epidemiology scandal --samplesize 1000 --compensate_samplesize
./train.py epidemiology scandal --samplesize 2000 --compensate_samplesize
./train.py epidemiology scandal --samplesize 5000 --compensate_samplesize
./train.py epidemiology scandal --samplesize 10000 --compensate_samplesize
./train.py epidemiology scandal --samplesize 20000 --compensate_samplesize
./train.py epidemiology scandal --samplesize 50000 --compensate_samplesize
./train.py epidemiology scandal --compensate_samplesize
