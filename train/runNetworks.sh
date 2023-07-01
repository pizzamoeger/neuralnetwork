#!/bin/bash

make

n=10  # Set the number of times to run the command

for ((i=1; i<=$n; i++))
do
  nohup ./neuralnetwork networks/net$i.txt > networks/net$i.out &
done

