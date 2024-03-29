runs_per_size=10
start_size=5
end_size=512
step_size=5
file_name="compare/CPU_n_squared.txt"

> $file_name

sed -i -e '/\/\/ FIND-TAG-STORING/{n; r /dev/stdin' -e 'N;N;d;}' train/main.cpp <<EOF
    // cerr << "Where should the network be stored? "; string filename; cin >> filename;
    // string filename = argv[1];
    // net.save(filename);
EOF

sed -i -e '/\/\/ FIND-TAG-EPOCHS/{n; r /dev/stdin' -e 'N;N;d;}' train/main.cpp <<EOF
    // cerr << "epochs: "; cin >> params.epochs;
    // params.epochs = 150;
    params.epochs = 0;
EOF

sed -i -e '/\/\/ FIND-TAG-OUTPUT/{n; r /dev/stdin' -e 'd;}' train/main.cpp <<EOF
    std::cout << durationTest << "\n";
EOF

for ((i=start_size; i<end_size; i+=step_size))
do

  sed -i -e '/\/\/ FIND-TAG-N/{n; r /dev/stdin' -e 'd;}' train/Network.h <<EOF
    #define NEURONS $i
EOF

  make

  timeTot=0
  for ((j=0; j<runs_per_size; j++))
  do
    time=$(./neuralnetwork)
    echo $time
    timeTot=$(echo "$timeTot+$time" | bc -l)
  done

  timeTot=$(echo "$timeTot/$runs_per_size" | bc -l)
  echo -e "$i: $timeTot" >> $file_name
done

sed -i -e '/\/\/ FIND-TAG-ARCHITECTURE/{n; r /dev/stdin' -e 'd;}' train/main.cpp <<EOF
    fully_connected2.n_out = {30, 1, 1};
EOF

sed -i -e '/\/\/ FIND-TAG-STORING/{n; r /dev/stdin' -e 'N;N;d;}' train/main.cpp <<EOF
    cerr << "Where should the network be stored? "; string filename; cin >> filename;
    // string filename = argv[1];
    net.save(filename);
EOF

sed -i -e '/\/\/ FIND-TAG-EPOCHS/{n; r /dev/stdin' -e 'N;N;d;}' train/main.cpp <<EOF
    cerr << "epochs: "; cin >> params.epochs;
    // params.epochs = 150;
    // params.epochs = 0;
EOF

sed -i -e '/\/\/ FIND-TAG-OUTPUT/{n; r /dev/stdin' -e 'd;}' train/main.cpp <<EOF
    // cout << evtst.second << "\n";
EOF