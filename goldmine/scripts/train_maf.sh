#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology maf --samplesize 100 --compensate_samplesize
./train.py epidemiology maf --samplesize 200 --compensate_samplesize
./train.py epidemiology maf --samplesize 500 --compensate_samplesize
./train.py epidemiology maf --samplesize 1000 --compensate_samplesize
./train.py epidemiology maf --samplesize 2000 --compensate_samplesize
./train.py epidemiology maf --samplesize 5000 --compensate_samplesize
./train.py epidemiology maf --samplesize 10000 --compensate_samplesize
./train.py epidemiology maf --samplesize 20000 --compensate_samplesize
./train.py epidemiology maf --samplesize 50000 --compensate_samplesize
./train.py epidemiology maf --compensate_samplesize
