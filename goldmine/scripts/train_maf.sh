#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology maf --samplesize 100
./train.py epidemiology maf --samplesize 200
./train.py epidemiology maf --samplesize 500
./train.py epidemiology maf --samplesize 1000
./train.py epidemiology maf --samplesize 2000
./train.py epidemiology maf --samplesize 5000
./train.py epidemiology maf --samplesize 10000
./train.py epidemiology maf --samplesize 20000
./train.py epidemiology maf --samplesize 50000
./train.py epidemiology maf
