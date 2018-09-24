#!/bin/bash

cd /Users/johannbrehmer/work/projects/scandal/goldmine/goldmine

job=0
while [ $job -le 9 ]
do
    echo ''
    echo ''
    echo ''
    echo "Starting job $job"
    echo ''

    ./test.py lotkavolterra maf -i $job --samplesize 1000 --classifiertest
    ./test.py lotkavolterra maf -i $job --samplesize 2000 --classifiertest
    ./test.py lotkavolterra maf -i $job --samplesize 5000 --classifiertest
    ./test.py lotkavolterra maf -i $job --samplesize 10000 --classifiertest
    ./test.py lotkavolterra maf -i $job --samplesize 20000 --classifiertest
    ./test.py lotkavolterra maf -i $job --samplesize 50000 --classifiertest
    ./test.py lotkavolterra maf -i $job --samplesize 100000 --classifiertest
    ./test.py lotkavolterra maf -i $job --samplesize 200000 --classifiertest
    #./test.py lotkavolterra maf -i $job --samplesize 500000 --classifiertest
    #./test.py lotkavolterra maf -i $job --samplesize 1000000 --classifiertest

    ((job++))
done

echo ''
echo ''
echo ''
echo 'All done'
echo ''