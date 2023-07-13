#!/bin/bash

# set epochs to 150
sed -i -e '/\/\/ FIND-TAG-EPOCHS/{n; r /dev/stdin' -e 'N;N;d;}' train/main.cpp <<EOF
    // cerr << "epochs: "; cin >> params.epochs;
    params.epochs = 150;
    // params.epochs = 5;
EOF

# storing as argument

sed -i -e '/\/\/ FIND-TAG-STORING/{n; r /dev/stdin' -e 'N;N;d;}' train/main.cpp <<EOF
    // cerr << "Where should the network be stored? "; string filename; cin >> filename;
    string filename = argv[1];
    net.save(filename);
EOF

make

# set epochs back to reading them from input

sed -i -e '/\/\/ FIND-TAG-EPOCHS/{n; r /dev/stdin' -e 'N;N;d;}' train/main.cpp <<EOF
    cerr << "epochs: "; cin >> params.epochs;
    // params.epochs = 150;
    // params.epochs = 5;
EOF

# storing as reading from input

sed -i -e '/\/\/ FIND-TAG-STORING/{n; r /dev/stdin' -e 'N;N;d;}' train/main.cpp <<EOF
    cerr << "Where should the network be stored? "; string filename; cin >> filename;
    // string filename = argv[1];
    net.save(filename);
EOF

n=5
# the program was run with "./runNetworks n" and this accesses the n
n="$1"

for ((i=0; i<n; i++))
do
  nohup ./neuralnetwork "networks/net$i.txt" > "networks/net$i.out" &
done

