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

    ./test.py lotkavolterra maf -i $job --samplesize 1000
    ./test.py lotkavolterra maf -i $job --samplesize 2000
    ./test.py lotkavolterra maf -i $job --samplesize 5000
    ./test.py lotkavolterra maf -i $job --samplesize 10000
    ./test.py lotkavolterra maf -i $job --samplesize 20000
    ./test.py lotkavolterra maf -i $job --samplesize 50000
    ./test.py lotkavolterra maf -i $job --samplesize 100000
    ./test.py lotkavolterra maf -i $job --samplesize 200000
    #./test.py lotkavolterra maf -i $job --samplesize 500000
    #./test.py lotkavolterra maf -i $job --samplesize 1000000

    ((job++))
done

echo ''
echo ''
echo ''
echo 'All done'
echo ''