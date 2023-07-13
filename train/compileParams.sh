# turn off the reduction of learning rates

sed -i -e '/\/\/ FIND-TAG-REDUCTION/{n; r /dev/stdin' -e 'N;N;N;d;}' main.cpp <<EOF
    params.fcBRed = 0;
    params.fcWRed = 0;
    params.convBRed = 0;
    params.convWRed = 0;
EOF

# set epochs to 5 epochs

sed -i -e '/\/\/ FIND-TAG-EPOCHS/{n; r /dev/stdin' -e 'N;N;d;}' main.cpp <<EOF
    // cerr << "epochs: "; cin >> params.epochs;
    // params.epochs = 150;
    params.epochs = 5;
EOF

# output the accuracy with cout

sed -i -e '/\/\/ FIND-TAG-OUTPUT/{n; r /dev/stdin' -e 'd;}' main.cpp <<EOF
    cout << (float)correctTest / params.test_data_size << "\n";
EOF

# no storing

sed -i -e '/\/\/ FIND-TAG-STORING/{n; r /dev/stdin' -e 'N;N;d;}' main.cpp <<EOF
    // cerr << "Where should the network be stored? "; string filename; cin >> filename;
    // string filename = argv[1];
    // net.save(filename);
EOF

# adjust layers to fully connected architecture

sed -i -e '/\/\/ FIND-TAG-LAYERS/{n; r /dev/stdin' -e 'N;N;N;N;N;N;N;d;}' main.cpp <<EOF
    int L = 4;
    layer_data* layers = new layer_data[L];
    layers[0] = input;
    layers[1] = convolutional;
    layers[1] = maxpool;
    layers[1] = fully_connected1;
    layers[2] = fully_connected2;
    layers[3] = outt;
EOF

g++ -std=c++17 -Wall -Wextra -O3 main.cpp Network.cpp layer.cpp misc.cpp -o fully_connected

# adjust layers to cnn

sed -i -e '/\/\/ FIND-TAG-LAYERS/{n; r /dev/stdin' -e 'N;N;N;N;N;N;N;d;}' main.cpp <<EOF
    int L = 6;
    layer_data* layers = new layer_data[L];
    layers[0] = input;
    layers[1] = convolutional;
    layers[2] = maxpool;
    layers[3] = fully_connected1;
    layers[4] = fully_connected2;
    layers[5] = outt;
EOF

g++ -std=c++17 -Wall -Wextra -O3 main.cpp Network.cpp layer.cpp misc.cpp -o cnn

# turn on reduction again

sed -i -e '/\/\/ FIND-TAG-REDUCTION/{n; r /dev/stdin' -e 'N;N;N;d;}' main.cpp <<EOF
    params.fcBRed = params.fully_connected_biases_learning_rate*99/10000;
    params.fcWRed = params.fully_connected_weights_learning_rate*99/10000;
    params.convBRed = params.convolutional_biases_learning_rate*99/10000;
    params.convWRed = params.convolutional_weights_learning_rate*99/10000;
EOF

# set epochs back to reading them from input

sed -i -e '/\/\/ FIND-TAG-EPOCHS/{n; r /dev/stdin' -e 'N;N;d;}' main.cpp <<EOF
    cerr << "epochs: "; cin >> params.epochs;
    // params.epochs = 150;
    // params.epochs = 5;
EOF

# dont output the accuracy with cout

sed -i -e '/\/\/ FIND-TAG-OUTPUT/{n; r /dev/stdin' -e 'd;}' main.cpp <<EOF
    // cout << (float)correctTest / params.test_data_size << "\n";
EOF

# storing as reading from input

sed -i -e '/\/\/ FIND-TAG-STORING/{n; r /dev/stdin' -e 'N;N;d;}' main.cpp <<EOF
    cerr << "Where should the network be stored? "; string filename; cin >> filename;
    // string filename = argv[1];
    net.save(filename);
EOF