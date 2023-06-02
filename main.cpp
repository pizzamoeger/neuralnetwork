#include "includes.h"
#define float float
using namespace std;

int main() {
    srand(time(NULL));
/*
    cout << "create new network? (1/0):";
    bool newNN; cin >> newNN;
    Network net;

    if (newNN) {
        int L; cout << "num of hidden layers: "; cin >> L;
        L += 2;
        vector<int> sizes (L); sizes[0] = 784; sizes[L-1] = 10;

        cout << "sizes of hidden layers: ";
        for (int i = 1; i < L-1; i++) cin >> sizes[i];

        net.init(sizes, relu, reluPrime, crossEntropyPrime);
    } else {
        string filename; cout << "filename: "; cin >> filename;
        vector<int> tmp = {};
        net.init(tmp, relu, reluPrime, crossEntropyPrime);
        net.load(filename);
    }*/

    Network net;

    layer_data input;
    input.type = 0;
    input.n_out = {28, 28, 1};

    layer_data convolutional;
    convolutional.type = 1;
    convolutional.n_in = input.n_out;
    convolutional.stride_length = 1;
    convolutional.receptive_field_length = 5;
    convolutional.activationFunctPrime = reluPrime;
    convolutional.activationFunct = relu;
    convolutional.n_out = {(convolutional.n_in.x-convolutional.receptive_field_length+1)/convolutional.stride_length, (convolutional.n_in.y-convolutional.receptive_field_length+1)/convolutional.stride_length, 3};

    layer_data c;
    c.type = 1;
    c.n_in = convolutional.n_out;
    c.stride_length = 1;
    c.receptive_field_length = 5;
    c.activationFunctPrime = reluPrime;
    c.activationFunct = relu;
    c.n_out = {(c.n_in.x-c.receptive_field_length+1)/c.stride_length, (c.n_in.y-c.receptive_field_length+1)/c.stride_length, 3};

    layer_data maxpool;
    maxpool.type = 2;
    maxpool.n_in = convolutional.n_out;
    maxpool.summarized_region_length = 2;
    maxpool.n_out = {maxpool.n_in.x/maxpool.summarized_region_length, maxpool.n_in.y/maxpool.summarized_region_length, maxpool.n_in.feature_maps};

    layer_data flatten;
    flatten.type = 3;
    flatten.n_in = maxpool.n_out;
    flatten.n_out = {flatten.n_in.x*flatten.n_in.y*flatten.n_in.feature_maps, 1, 1};

    layer_data fully_connected1;
    fully_connected1.type = 4;
    fully_connected1.n_in = flatten.n_out;
    fully_connected1.activationFunctPrime = reluPrime;
    fully_connected1.activationFunct = relu;
    fully_connected1.n_out = {30, 1, 1};

    layer_data fully_connected2;
    fully_connected2.type = 4;
    fully_connected2.activationFunctPrime = reluPrime;
    fully_connected2.activationFunct = relu;
    fully_connected2.n_in = fully_connected1.n_out;
    fully_connected2.n_out = {30, 1, 1};

    layer_data outt;
    outt.type = 4;
    outt.activationFunctPrime = reluPrime;
    outt.activationFunct = relu;
    outt.n_in = fully_connected2.n_out;
    outt.last_layer = true;

    outt.n_out = {10, 1, 1};

    vector layers = {input, convolutional, maxpool, flatten, fully_connected1, fully_connected2, outt};
    net.init(layers, crossEntropyPrime);

    // train network
    auto test_data = load_data("mnist_test_normalized.data");
    auto training_data = load_data("mnist_train_normalized.data");

    auto params = get_params();
    params.test_data_size  = test_data.size();
    params.training_data_size = training_data.size();

    net.SGD(training_data, test_data, params);

    /*bool save; cout << "save network? (1/0):"; cin >> save;
    if (save) {
        string filename; cout << "filename: "; cin >> filename;
        net.save(filename);
    }*/

    auto [correctTrain, durationTrain] = net.evaluate(training_data, params);
    auto [correctTest, durationTest] = net.evaluate(test_data, params);

    cout << "accuracy in training data: " << (float)correctTrain / params.training_data_size << "\n";
    cout << "general accuracy: " << (float)correctTest / params.test_data_size << "\n";
}