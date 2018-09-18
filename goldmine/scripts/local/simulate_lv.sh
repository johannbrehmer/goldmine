#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

# ./simulate.py --nthetas 10 --nsamples 10 --noratio lotkavolterra train
./simulate.py --singletheta --nsamples 100 --noratio lotkavolterra small

#mprof run python -m memory_profiler ./simulate.py --singletheta --nsamples 100 --noratio lotkavolterra test
#mprof plot
