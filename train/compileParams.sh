sed -i -e '/\/\/ FIND-TAG-231/{n; r /dev/stdin' -e 'N;N;N;N;N;N;N;d;}' main.cpp <<EOF
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

sed -i -e '/\/\/ FIND-TAG-231/{n; r /dev/stdin' -e 'N;N;N;N;N;N;N;d;}' main.cpp <<EOF
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
