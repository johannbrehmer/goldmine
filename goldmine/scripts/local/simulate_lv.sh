#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

python -m memory_profiler ./simulate.py --nthetas 10 --nsamples 10 --noratio lotkavolterra train
# ./simulate.py --singletheta --nsamples 100000 --noratio lotkavolterra test
