#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./test.py epidemiology maf --trainingsamplesize 100 --classifiertest
./test.py epidemiology maf --trainingsamplesize 200 --classifiertest
./test.py epidemiology maf --trainingsamplesize 500 --classifiertest
./test.py epidemiology maf --trainingsamplesize 1000 --classifiertest
./test.py epidemiology maf --trainingsamplesize 2000 --classifiertest
./test.py epidemiology maf --trainingsamplesize 5000 --classifiertest
./test.py epidemiology maf --trainingsamplesize 10000 --classifiertest
./test.py epidemiology maf --trainingsamplesize 20000 --classifiertest
./test.py epidemiology maf --trainingsamplesize 50000 --classifiertest
./test.py epidemiology maf --classifiertest
