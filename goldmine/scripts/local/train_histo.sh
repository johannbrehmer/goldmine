#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology2d histogram --samplesize 100 --singletheta  --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 200 --singletheta  --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 500 --singletheta  --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 1000 --singletheta  --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 2000 --singletheta  --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 5000 --singletheta  --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 10000 --singletheta  --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 20000 --singletheta  --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 50000 --singletheta  --xhistos1d --xbins 10
./train.py epidemiology2d histogram --singletheta  --xhistos1d --xbins 10
