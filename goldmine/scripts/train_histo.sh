#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology histogram --samplesize 100 --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 200 --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 500 --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 1000 --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 2000 --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 5000 --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 10000 --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 20000 --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 50000 --fillemptybins --xhistos1d
./train.py epidemiology histogram --fillemptybins --xhistos1d

./train.py epidemiology histogram --samplesize 100 --singletheta --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 200 --singletheta --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 500 --singletheta --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 1000 --singletheta --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 2000 --singletheta --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 5000 --singletheta --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 10000 --singletheta --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 20000 --singletheta --fillemptybins --xhistos1d
./train.py epidemiology histogram --samplesize 50000 --singletheta --fillemptybins --xhistos1d
./train.py epidemiology histogram --singletheta --fillemptybins --xhistos1d
