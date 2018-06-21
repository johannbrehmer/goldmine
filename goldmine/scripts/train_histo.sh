#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology histogram --samplesize 100 --fillemptybins
./train.py epidemiology histogram --samplesize 200 --fillemptybins
./train.py epidemiology histogram --samplesize 500 --fillemptybins
./train.py epidemiology histogram --samplesize 1000 --fillemptybins
./train.py epidemiology histogram --samplesize 2000 --fillemptybins
./train.py epidemiology histogram --samplesize 5000 --fillemptybins
./train.py epidemiology histogram --samplesize 10000 --fillemptybins
./train.py epidemiology histogram --samplesize 20000 --fillemptybins
./train.py epidemiology histogram --samplesize 50000 --fillemptybins
./train.py epidemiology histogram --fillemptybins
