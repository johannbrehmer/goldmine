#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology histogram --samplesize 100
./train.py epidemiology histogram --samplesize 200
./train.py epidemiology histogram --samplesize 500
./train.py epidemiology histogram --samplesize 1000
./train.py epidemiology histogram --samplesize 2000
./train.py epidemiology histogram --samplesize 5000
./train.py epidemiology histogram --samplesize 10000
./train.py epidemiology histogram --samplesize 20000
./train.py epidemiology histogram --samplesize 50000
./train.py epidemiology histogram
