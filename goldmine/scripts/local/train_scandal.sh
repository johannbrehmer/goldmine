#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

./train.py epidemiology2d scandal --samplesize 100  
./train.py epidemiology2d scandal --samplesize 200  
./train.py epidemiology2d scandal --samplesize 500  
./train.py epidemiology2d scandal --samplesize 1000  
./train.py epidemiology2d scandal --samplesize 2000  
./train.py epidemiology2d scandal --samplesize 5000  
./train.py epidemiology2d scandal --samplesize 10000  
./train.py epidemiology2d scandal --samplesize 20000  
./train.py epidemiology2d scandal --samplesize 50000  
./train.py epidemiology2d scandal  
