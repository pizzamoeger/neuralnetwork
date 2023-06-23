#include "includes.h"
using namespace std;

int main() {
    //
    srand(time(NULL));

    Network net;

    layer_data input;
    input.type = LAYER_NUM_INPUT;
    input.n_out = {28, 28, 1};

    layer_data convolutional;
    convolutional.type = LAYER_NUM_CONVOLUTIONAL;
    convolutional.stride_length = 1;
    convolutional.receptive_field_length = 5;
    convolutional.activationFunctPrime = reluPrime;
    convolutional.activationFunct = relu;
    convolutional.n_out = {-1,-1, 3};

    layer_data maxpool;
    maxpool.type = 2;
    maxpool.summarized_region_length = 2;

    layer_data fully_connected1;
    fully_connected1.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected1.activationFunctPrime = reluPrime;
    fully_connected1.activationFunct = relu;
    fully_connected1.n_out = {30, 1, 1};

    layer_data fully_connected2;
    fully_connected2.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected2.activationFunctPrime = reluPrime;
    fully_connected2.activationFunct = relu;
    fully_connected2.n_out = {30, 1, 1};

    layer_data outt;
    outt.type = LAYER_NUM_FULLY_CONNECTED;
    outt.activationFunctPrime = reluPrime;
    outt.activationFunct = relu;
    outt.last_layer = true;
    outt.n_out = {10, 1, 1};

    int L = 6;
    layer_data* layers = new layer_data[L];
    layers[0] = input;
    layers[1] = convolutional;
    layers[2] = maxpool;
    layers[3] = fully_connected1;
    layers[4] = fully_connected2;
    layers[5] = outt;
    net.init(layers, L, crossEntropyPrime);

    // train network
    auto [test_data, test_data_size] = load_data("mnist_test_normalized.data");
    auto [training_data, training_data_size] = load_data("mnist_train_normalized.data");

    auto params = get_params();
    params.test_data_size  = test_data_size;
    params.training_data_size = training_data_size;

    net.SGD(training_data, test_data, params);

    auto [correctTest, durationTest] = net.evaluate(test_data, test_data_size);
    auto [correctTrain, durationTrain] = net.evaluate(training_data, training_data_size);

    cout << "accuracy in training data: " << (float)correctTrain / params.training_data_size << "\n";
    cout << "general accuracy: " << (float)correctTest / params.test_data_size << "\n";

    net.save("net.txt");

    clear_data(test_data);
    clear_data(training_data);
    net.clear();
    delete[] layers;
}