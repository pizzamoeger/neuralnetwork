#!/bin/bash

make

n=10  # Set the number of times to run the command

for ((i=1; i<=$n; i++))
do
  nohup ./neuralnetwork net$i.txt > net$i.out &
done

