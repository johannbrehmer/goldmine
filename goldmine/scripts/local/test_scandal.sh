#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./test.py epidemiology2d scandal --samplesize 100 --classifiertest
./test.py epidemiology2d scandal --samplesize 200 --classifiertest
./test.py epidemiology2d scandal --samplesize 500 --classifiertest
./test.py epidemiology2d scandal --samplesize 1000 --classifiertest
./test.py epidemiology2d scandal --samplesize 2000 --classifiertest
./test.py epidemiology2d scandal --samplesize 5000 --classifiertest
./test.py epidemiology2d scandal --samplesize 10000 --classifiertest
./test.py epidemiology2d scandal --samplesize 20000 --classifiertest
./test.py epidemiology2d scandal --samplesize 50000 --classifiertest
./test.py epidemiology2d scandal --classifiertest
