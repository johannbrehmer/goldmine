#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./test.py epidemiology maf --trainingsamplesize 100 
./test.py epidemiology maf --trainingsamplesize 200 
./test.py epidemiology maf --trainingsamplesize 500 
./test.py epidemiology maf --trainingsamplesize 1000 
./test.py epidemiology maf --trainingsamplesize 2000 
./test.py epidemiology maf --trainingsamplesize 5000 
./test.py epidemiology maf --trainingsamplesize 10000 
./test.py epidemiology maf --trainingsamplesize 20000 
./test.py epidemiology maf --trainingsamplesize 50000 
./test.py epidemiology maf 
