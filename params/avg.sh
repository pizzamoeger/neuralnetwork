#!/bin/bash

declare -a params
params_size=6
for ((i=0; i<params_size; i++))
do
  params[i]=0
done

n=10  # Set the number of times to run the command
n="$1"

for ((i=0; i<n; i++))
do
  file="params$i.out"
  # Check if the file exists
  if [ -e "$file" ]; then
      # Extract the last line containing the numbers
      last_line=$(tail -n 1 "$file")

      # Use grep to extract the numbers between brackets
      numbers=$(echo "$last_line" | grep -o '\[.*\]')

      # Remove brackets and split the numbers into an array
      IFS=', ' read -r -a add_params <<< "${numbers:1:${#numbers}-2}"

      # Process each number in the array
      for ((j=0; j<params_size; j++))
      do
          params[j]=$(echo "${add_params[j]}+${params[j]}" | bc -l)
      done
  else
      echo "File not found: $file"
  fi
done

for ((i=0; i<params_size; i++))
do
  params[i]=$(echo "${params[i]}/${n}" | bc -l)
done
echo "${params[@]}"
