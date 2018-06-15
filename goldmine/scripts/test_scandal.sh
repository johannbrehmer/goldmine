#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 100 --classifiertest
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 200 --classifiertest
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 500 --classifiertest
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 1000 --classifiertest
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 2000 --classifiertest
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 5000 --classifiertest
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 10000 --classifiertest
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 20000 --classifiertest
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 50000 --classifiertest
./test.py epidemiology scandal --alpha 0.01 --classifiertest
