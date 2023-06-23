#include "includes.h"
using namespace std;

int main(int argc, char* argv[]) {

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

    /*
    layer_data input;
    input.type = LAYER_NUM_INPUT;
    input.n_out = {2, 2, 1};

    layer_data convolutional;
    convolutional.type = LAYER_NUM_CONVOLUTIONAL;
    convolutional.stride_length = 1;
    convolutional.receptive_field_length = 2;
    convolutional.activationFunctPrime = reluPrime;
    convolutional.activationFunct = relu;
    convolutional.n_out = {1,1, 1};

    layer_data fully_connected;
    fully_connected.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected.activationFunctPrime = reluPrime;
    fully_connected.activationFunct = relu;
    fully_connected.n_out = {1, 1, 1};*/

    vector layers = {input, fully_connected1, outt};
    net.init(layers, crossEntropyPrime);

    //net.save("nettibetti.txt");


    // train network
    auto test_data = load_data("mnist_test_normalized.data");
    auto training_data = load_data("mnist_train_normalized.data");

    auto params = get_params();
    assert(argc == 7);

    params.fully_connected_weights_learning_rate = stof(argv[1]);
    params.fully_connected_biases_learning_rate = stof(argv[2]);
    params.convolutional_weights_learning_rate = stof(argv[3]);
    params.convolutional_biases_learning_rate = stof(argv[4]);
    params.L2_regularization_term = stof(argv[5]);
    params.momentum_coefficient = stof(argv[6]);

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

    cerr << "accuracy in training data: " << (float)correctTrain / params.training_data_size << "\n";
    cerr << "general accuracy: " << (float)correctTest / params.test_data_size << "\n";
    cout << (float) correctTest / params.test_data_size;
    /*vector<float> inputt = {0, 0.25, 0.5, 0.75};
    auto pred = net.feedforward(inputt);
    cout << "\n";*/
    net.save("net.txt");
}