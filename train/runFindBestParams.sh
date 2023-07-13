(
  cd "train" || exit
  ./compileParams.sh
)

n=10  # Set the number of times to run the command

for ((i=0; i<n; i++))
do
  nohup echo "params/params$i.txt" | python3 train/findBestParams.py > "params/params$i.out" &
done