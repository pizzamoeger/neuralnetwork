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

        net.init(sizes, sigmoid, sigmoidPrime, crossEntropyPrime);
    } else {
        string filename; cout << "filename: "; cin >> filename;
        vector<int> tmp = {};
        net.init(tmp, sigmoid, sigmoidPrime, crossEntropyPrime);
        net.load(filename);
    }*/

    Network net;

    layer_data input;
    input.type = 0;
    input.n_out = {28, 28};

    layer_data convolutional;
    convolutional.type = 1;
    convolutional.n_in = input.n_out;
    convolutional.stride_length = 1;
    convolutional.receptive_field_length = 5;
    convolutional.feature_maps = 5;
    convolutional.n_out = {(convolutional.n_in.x-convolutional.receptive_field_length+1)/convolutional.stride_length, (convolutional.n_in.y-convolutional.receptive_field_length+1)/convolutional.stride_length};

    layer_data maxpool;
    maxpool.type = 2;
    maxpool.n_in = convolutional.n_out;
    maxpool.summarized_region_length = 2;
    maxpool.n_out = {maxpool.n_in.x/maxpool.summarized_region_length, maxpool.n_in.y/maxpool.summarized_region_length};

    layer_data flatten;
    flatten.type = 3;
    flatten.feature_maps = 1; // all feature maps before multiplied
    flatten.n_in = maxpool.n_out;
    flatten.n_out = {flatten.n_in.x*flatten.n_in.y, 1};

    layer_data fully_connected1;
    fully_connected1.type = 4;
    fully_connected1.n_in = flatten.n_out;
    fully_connected1.n_out = {30, 1};

    layer_data fully_connected2;
    fully_connected2.type = 4;
    fully_connected2.n_in = fully_connected1.n_out;
    fully_connected2.n_out = {30, 1};

    layer_data outt;
    outt.type = 4;
    outt.n_in = fully_connected1.n_out;
    outt.n_out = {10, 1};

    vector<layer_data> layers = {input, convolutional, maxpool, flatten, fully_connected1, outt};
    net.init(layers, sigmoid, sigmoidPrime, crossEntropyPrime);

    // train network
    auto test_data = load_data("mnist_test_normalized.data");
    //cout << "train network? (1/0):";
    bool train=1; //cin >> train;

    if (train) {
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

        int correct = 0;
        for (int k = 0; k < training_data.size(); k++) {
            vector<float> output = net.feedforward(training_data[k].first).first[net.L-1][0][0];
            int max = 0;
            for (int j = 0; j < output.size(); j++) {
                if (output[j] > output[max]) max = j;
            }
            if (training_data[k].second[max] == 1) correct++;
        }
        cout << "accuracy in training data: " << (float)correct / training_data.size() << "\n";
    }

    // test network
    int correct = 0;
    for (int k = 0; k < test_data.size(); k++) {
        vector<float> output = net.feedforward(test_data[k].first).first[net.L-1][0][0];
        int max = 0;
        for (int j = 0; j < output.size(); j++) {
            if (output[j] > output[max]) max = j;
        }
        if (test_data[k].second[max] == 1) correct++;
    }
    cout << "general accuracy: " << (float)correct / test_data.size() << "\n";
}