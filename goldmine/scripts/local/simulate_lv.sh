#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./simulate.py --nthetas 10000 --nsamples 100 --noratio lotkavolterra train
./simulate.py --singletheta --nsamples 100000 --noratio lotkavolterra test
