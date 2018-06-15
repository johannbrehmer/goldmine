#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology maf --trainingsamplesize 100
./train.py epidemiology maf --trainingsamplesize 200
./train.py epidemiology maf --trainingsamplesize 500
./train.py epidemiology maf --trainingsamplesize 1000
./train.py epidemiology maf --trainingsamplesize 2000
./train.py epidemiology maf --trainingsamplesize 5000
./train.py epidemiology maf --trainingsamplesize 10000
./train.py epidemiology maf --trainingsamplesize 20000
./train.py epidemiology maf --trainingsamplesize 50000
./train.py epidemiology maf
