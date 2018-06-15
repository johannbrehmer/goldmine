#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology scandal --alpha 0.01 --trainingsamplesize 100
./train.py epidemiology scandal --alpha 0.01 --trainingsamplesize 200
./train.py epidemiology scandal --alpha 0.01 --trainingsamplesize 500
./train.py epidemiology scandal --alpha 0.01 --trainingsamplesize 1000
./train.py epidemiology scandal --alpha 0.01 --trainingsamplesize 2000
./train.py epidemiology scandal --alpha 0.01 --trainingsamplesize 5000
./train.py epidemiology scandal --alpha 0.01 --trainingsamplesize 10000
./train.py epidemiology scandal --alpha 0.01 --trainingsamplesize 20000
./train.py epidemiology scandal --alpha 0.01 --trainingsamplesize 50000
./train.py epidemiology scandal --alpha 0.01
