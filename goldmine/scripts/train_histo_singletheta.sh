#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology histogram --samplesize 100 --singletheta --thetabins 1 --fillemptybins
./train.py epidemiology histogram --samplesize 200 --singletheta --thetabins 1 --fillemptybins
./train.py epidemiology histogram --samplesize 500 --singletheta --thetabins 1 --fillemptybins
./train.py epidemiology histogram --samplesize 1000 --singletheta --thetabins 1 --fillemptybins
./train.py epidemiology histogram --samplesize 2000 --singletheta --thetabins 1 --fillemptybins
./train.py epidemiology histogram --samplesize 5000 --singletheta --thetabins 1 --fillemptybins
./train.py epidemiology histogram --samplesize 10000 --singletheta --thetabins 1 --fillemptybins
./train.py epidemiology histogram --samplesize 20000 --singletheta --thetabins 1 --fillemptybins
./train.py epidemiology histogram --samplesize 50000 --singletheta --thetabins 1 --fillemptybins
./train.py epidemiology histogram --singletheta --thetabins 1 --fillemptybins
