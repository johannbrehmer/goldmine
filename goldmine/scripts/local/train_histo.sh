#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology2d histogram --samplesize 100 --singletheta  --xbins 20 --xhistos1d
./train.py epidemiology2d histogram --samplesize 200 --singletheta   --xbins 20 --xhistos1d
./train.py epidemiology2d histogram --samplesize 500 --singletheta   --xbins 20 --xhistos1d
./train.py epidemiology2d histogram --samplesize 1000 --singletheta   --xbins 20 --xhistos1d
./train.py epidemiology2d histogram --samplesize 2000 --singletheta   --xbins 20 --xhistos1d
./train.py epidemiology2d histogram --samplesize 5000 --singletheta   --xbins 20 --xhistos1d
./train.py epidemiology2d histogram --samplesize 10000 --singletheta   --xbins 20 --xhistos1d
./train.py epidemiology2d histogram --samplesize 20000 --singletheta   --xbins 20 --xhistos1d
./train.py epidemiology2d histogram --samplesize 50000 --singletheta   --xbins 20 --xhistos1d
./train.py epidemiology2d histogram --singletheta   --xbins 20 --xhistos1d
