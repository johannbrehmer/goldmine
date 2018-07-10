#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./test.py epidemiology2d maf --samplesize 100 --classifiertest
./test.py epidemiology2d maf --samplesize 200 --classifiertest
./test.py epidemiology2d maf --samplesize 500 --classifiertest
./test.py epidemiology2d maf --samplesize 1000 --classifiertest
./test.py epidemiology2d maf --samplesize 2000 --classifiertest
./test.py epidemiology2d maf --samplesize 5000 --classifiertest
./test.py epidemiology2d maf --samplesize 10000 --classifiertest
./test.py epidemiology2d maf --samplesize 20000 --classifiertest
./test.py epidemiology2d maf --samplesize 50000 --classifiertest
./test.py epidemiology2d maf --classifiertest
