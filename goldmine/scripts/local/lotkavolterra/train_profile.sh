#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

python -m cProfile -s time ./train.py lotkavolterra scandal -i 99 --trainsample train_focus --samplesize 50000 --modellabel profile --epochs 1 --batchsize 512
