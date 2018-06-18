#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology maf --samplesize 100 --epochs 10 --compensate_samplesize
./train.py epidemiology maf --samplesize 200 --epochs 10 --compensate_samplesize
./train.py epidemiology maf --samplesize 500 --epochs 10 --compensate_samplesize
./train.py epidemiology maf --samplesize 1000 --epochs 10 --compensate_samplesize
./train.py epidemiology maf --samplesize 2000 --epochs 10 --compensate_samplesize
./train.py epidemiology maf --samplesize 5000 --epochs 10 --compensate_samplesize
./train.py epidemiology maf --samplesize 10000 --epochs 10 --compensate_samplesize
./train.py epidemiology maf --samplesize 20000 --epochs 10 --compensate_samplesize
./train.py epidemiology maf --samplesize 50000 --epochs 10 --compensate_samplesize
./train.py epidemiology maf --epochs 10 --compensate_samplesize
