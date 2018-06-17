#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 100 
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 200 
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 500 
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 1000 
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 2000 
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 5000 
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 10000 
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 20000 
./test.py epidemiology scandal --alpha 0.01 --trainingsamplesize 50000 
./test.py epidemiology scandal --alpha 0.01 
