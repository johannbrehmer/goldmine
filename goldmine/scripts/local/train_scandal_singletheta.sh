#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology2d scandal --samplesize 100 --singletheta
./train.py epidemiology2d scandal --samplesize 200 --singletheta
./train.py epidemiology2d scandal --samplesize 500 --singletheta
./train.py epidemiology2d scandal --samplesize 1000 --singletheta
./train.py epidemiology2d scandal --samplesize 2000 --singletheta
./train.py epidemiology2d scandal --samplesize 5000 --singletheta
./train.py epidemiology2d scandal --samplesize 10000 --singletheta
./train.py epidemiology2d scandal --samplesize 20000 --singletheta
./train.py epidemiology2d scandal --samplesize 50000 --singletheta
./train.py epidemiology2d scandal --singletheta
