#!/bin/bash
set -e
#for i in {0..100}; do 
#	nohup ./run.sh 0 > 0.log 2>&1 &
#	./run.sh 1
#done

nohup ./run.sh 0 > 0.log 2>&1 &
nohup ./run.sh 1 > 1.log 2>&1 &
