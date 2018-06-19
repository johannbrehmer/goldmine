#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology histogram --samplesize 100 --compensate_samplesize
./train.py epidemiology histogram --samplesize 200 --compensate_samplesize
./train.py epidemiology histogram --samplesize 500 --compensate_samplesize
./train.py epidemiology histogram --samplesize 1000 --compensate_samplesize
./train.py epidemiology histogram --samplesize 2000 --compensate_samplesize
./train.py epidemiology histogram --samplesize 5000 --compensate_samplesize
./train.py epidemiology histogram --samplesize 10000 --compensate_samplesize
./train.py epidemiology histogram --samplesize 20000 --compensate_samplesize
./train.py epidemiology histogram --samplesize 50000 --compensate_samplesize
./train.py epidemiology histogram --compensate_samplesize
