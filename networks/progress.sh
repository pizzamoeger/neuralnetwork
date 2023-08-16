start=$1
end=$2
n=$(echo "$end-$start" | bc -l)

maxVal=0.0
netWithMax=0
for ((i=0; i<10; i=0))
do
  declare -a output
  cnt=0
  for ((j=0; j<n; j++))
  do
    ind=$(echo "$j+$start" | bc -l)
    both=$(tail -n 2 "net${ind}.out")
    secondToLast=$(echo -e "$both" | grep -o "^.*A.*s")
    output+="net${ind}.out: $secondToLast\n"

    isFinished=$(echo -e "$both" | grep -o "DONE")

    if [ -n "$isFinished" ]; then
      cnt=$(echo "$cnt+1" | bc -l)
      val=$(echo -e "$both" | grep -oE "0\.[0-9]+")
      netWithMax=$(echo "if ($val > $maxVal) ${ind} else $netWithMax" | bc -l)
      maxVal=$(echo "if ($val > $maxVal) $val else $maxVal" | bc -l)
    fi
  done
  clear
  echo -e "$output"
  output=""
  echo $cnt
  echo $n
  if [ $(echo "$cnt == $end" | bc -l) ] ; then
    clear
    echo "net$netWithMax.txt is the best network"
    echo "general accuracy: $maxVal"
    break
  fi
done
