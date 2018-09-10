#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

counter=1
while [ $counter -lt 10 ]
do
  ./test.py epidemiology2d scandal --samplesize 100 --classifiertest -i $counter
  ./test.py epidemiology2d scandal --samplesize 200 --classifiertest -i $counter
  ./test.py epidemiology2d scandal --samplesize 500 --classifiertest -i $counter
  ./test.py epidemiology2d scandal --samplesize 1000 --classifiertest -i $counter
  ./test.py epidemiology2d scandal --samplesize 2000 --classifiertest -i $counter
  ./test.py epidemiology2d scandal --samplesize 5000 --classifiertest -i $counter
  ./test.py epidemiology2d scandal --samplesize 10000 --classifiertest -i $counter
  ./test.py epidemiology2d scandal --samplesize 20000 --classifiertest -i $counter
  ./test.py epidemiology2d scandal --samplesize 50000 --classifiertest -i $counter
  ./test.py epidemiology2d scandal --classifiertest -i $counter

  ((counter++))
done
