#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

counter=1
while [ $counter -lt 10 ]
do
  ./test.py epidemiology2d maf --samplesize 100 --classifiertest -i $counter
  ./test.py epidemiology2d maf --samplesize 200 --classifiertest -i $counter
  ./test.py epidemiology2d maf --samplesize 500 --classifiertest -i $counter
  ./test.py epidemiology2d maf --samplesize 1000 --classifiertest -i $counter
  ./test.py epidemiology2d maf --samplesize 2000 --classifiertest -i $counter
  ./test.py epidemiology2d maf --samplesize 5000 --classifiertest -i $counter
  ./test.py epidemiology2d maf --samplesize 10000 --classifiertest -i $counter
  ./test.py epidemiology2d maf --samplesize 20000 --classifiertest -i $counter
  ./test.py epidemiology2d maf --samplesize 50000 --classifiertest -i $counter
  ./test.py epidemiology2d maf --classifiertest -i $counter

  ((counter++))
done
