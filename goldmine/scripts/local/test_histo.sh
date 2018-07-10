#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./test.py epidemiology2d histogram --samplesize 100 --singletheta
./test.py epidemiology2d histogram --samplesize 200 --singletheta
./test.py epidemiology2d histogram --samplesize 500 --singletheta
./test.py epidemiology2d histogram --samplesize 1000 --singletheta
./test.py epidemiology2d histogram --samplesize 2000 --singletheta
./test.py epidemiology2d histogram --samplesize 5000 --singletheta
./test.py epidemiology2d histogram --samplesize 10000 --singletheta
./test.py epidemiology2d histogram --samplesize 20000 --singletheta
./test.py epidemiology2d histogram --samplesize 50000 --singletheta
./test.py epidemiology2d histogram --singletheta
