#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology maf --samplesize 100 --singletheta
./train.py epidemiology maf --samplesize 200 --singletheta
./train.py epidemiology maf --samplesize 500 --singletheta
./train.py epidemiology maf --samplesize 1000 --singletheta
./train.py epidemiology maf --samplesize 2000 --singletheta
./train.py epidemiology maf --samplesize 5000 --singletheta
./train.py epidemiology maf --samplesize 10000 --singletheta
./train.py epidemiology maf --samplesize 20000 --singletheta
./train.py epidemiology maf --samplesize 50000 --singletheta
./train.py epidemiology maf --singletheta
