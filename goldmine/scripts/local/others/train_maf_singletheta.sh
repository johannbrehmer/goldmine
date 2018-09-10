#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology2d maf --samplesize 100 --singletheta
./train.py epidemiology2d maf --samplesize 200 --singletheta
./train.py epidemiology2d maf --samplesize 500 --singletheta
./train.py epidemiology2d maf --samplesize 1000 --singletheta
./train.py epidemiology2d maf --samplesize 2000 --singletheta
./train.py epidemiology2d maf --samplesize 5000 --singletheta
./train.py epidemiology2d maf --samplesize 10000 --singletheta
./train.py epidemiology2d maf --samplesize 20000 --singletheta
./train.py epidemiology2d maf --samplesize 50000 --singletheta
./train.py epidemiology2d maf --singletheta
