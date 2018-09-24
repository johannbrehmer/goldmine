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

    ./test.py lotkavolterra maf -i $job --samplesize 1000 --score
    ./test.py lotkavolterra maf -i $job --samplesize 2000 --score
    ./test.py lotkavolterra maf -i $job --samplesize 5000 --score
    ./test.py lotkavolterra maf -i $job --samplesize 10000 --score
    ./test.py lotkavolterra maf -i $job --samplesize 20000 --score
    ./test.py lotkavolterra maf -i $job --samplesize 50000 --score
    ./test.py lotkavolterra maf -i $job --samplesize 100000 --score
    ./test.py lotkavolterra maf -i $job --samplesize 200000 --score
    #./test.py lotkavolterra maf -i $job --samplesize 500000 --score
    #./test.py lotkavolterra maf -i $job --samplesize 1000000 --score

    ((job++))
done

echo ''
echo ''
echo ''
echo 'All done'
echo ''