#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology2d histogram --samplesize 100 --fillemptybins --xhistos1d --xbins 10 --thetabins 3
./train.py epidemiology2d histogram --samplesize 200 --fillemptybins --xhistos1d --xbins 10 --thetabins 3
./train.py epidemiology2d histogram --samplesize 500 --fillemptybins --xhistos1d --xbins 10 --thetabins 3
./train.py epidemiology2d histogram --samplesize 1000 --fillemptybins --xhistos1d --xbins 10 --thetabins 3
./train.py epidemiology2d histogram --samplesize 2000 --fillemptybins --xhistos1d --xbins 10 --thetabins 3
./train.py epidemiology2d histogram --samplesize 5000 --fillemptybins --xhistos1d --xbins 10 --thetabins 3
./train.py epidemiology2d histogram --samplesize 10000 --fillemptybins --xhistos1d --xbins 10 --thetabins 3
./train.py epidemiology2d histogram --samplesize 20000 --fillemptybins --xhistos1d --xbins 10 --thetabins 3
./train.py epidemiology2d histogram --samplesize 50000 --fillemptybins --xhistos1d --xbins 10 --thetabins 3
./train.py epidemiology2d histogram --fillemptybins --xhistos1d --xbins 10 --thetabins 3

./train.py epidemiology2d histogram --samplesize 100 --singletheta --fillemptybins --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 200 --singletheta --fillemptybins --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 500 --singletheta --fillemptybins --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 1000 --singletheta --fillemptybins --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 2000 --singletheta --fillemptybins --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 5000 --singletheta --fillemptybins --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 10000 --singletheta --fillemptybins --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 20000 --singletheta --fillemptybins --xhistos1d --xbins 10
./train.py epidemiology2d histogram --samplesize 50000 --singletheta --fillemptybins --xhistos1d --xbins 10
./train.py epidemiology2d histogram --singletheta --fillemptybins --xhistos1d --xbins 10
