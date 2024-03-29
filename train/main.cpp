#include "includes.h"
using namespace std;

int main(int argc, char** argv) {
    srand(time(NULL));

    Network net;

    layer_data input;
    input.type = LAYER_NUM_INPUT;
    //input.n_out = {28, 28, 1};
    input.n_out = {NEURONS, 1, 1};

    layer_data convolutional;
    convolutional.type = LAYER_NUM_CONVOLUTIONAL;
    convolutional.stride_length = 1;
    convolutional.receptive_field_length = 5;
    convolutional.activation_function = RELU;
    convolutional.n_out = {-1,-1, 3};

    layer_data maxpool;
    maxpool.type = 2;
    maxpool.summarized_region_length = 2;

    layer_data fully_connected1;
    fully_connected1.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected1.activation_function = RELU;
    fully_connected1.n_out = {30, 1, 1};

    layer_data fully_connected2;
    fully_connected2.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected2.activation_function = RELU;
    // FIND-TAG-ARCHITECTURE
    fully_connected2.n_out = {30, 1, 1};

    layer_data outt;
    outt.type = LAYER_NUM_FULLY_CONNECTED;
    outt.activation_function = RELU;
    outt.last_layer = true;
    //outt.n_out = {OUTPUT_NEURONS, 1, 1};
    outt.n_out = {NEURONS, 1, 1};


    // FIND-TAG-LAYERS
    int L = 2;
    layer_data* layers = new layer_data[L];
    layers[0] = input;
    layers[1] = convolutional;
    layers[1] = maxpool;
    layers[1] = fully_connected2;
    layers[1] = fully_connected2;
    layers[1] = outt;

    // train network
    auto [test_data, test_data_size] = load_data("mnist_test_normalized.data");
    auto [training_data, training_data_size] = load_data("mnist_train_normalized.data");

    auto params = get_params();
    if (argc == 7) {
        params.fully_connected_weights_learning_rate = atof(argv[1]);
        params.fully_connected_biases_learning_rate = atof(argv[2]);
        params.convolutional_weights_learning_rate = atof(argv[3]);
        params.convolutional_biases_learning_rate = atof(argv[4]);
        params.L2_regularization_term = atof(argv[5]);
        params.momentum_coefficient = atof(argv[6]);
    }
    params.test_data_size = test_data_size;
    params.training_data_size = training_data_size;

    // initialize params learning rate reduction
    // FIND-TAG-REDUCTION
    params.fcBRed = params.fully_connected_biases_learning_rate*99/10000;
    params.fcWRed = params.fully_connected_weights_learning_rate*99/10000;
    params.convBRed = params.convolutional_biases_learning_rate*99/10000;
    params.convWRed = params.convolutional_weights_learning_rate*99/10000;

    // FIND-TAG-EPOCHS
    cerr << "epochs: "; cin >> params.epochs;
    // params.epochs = 150;
    // params.epochs = 0;

    net.init(layers, L, crossEntropyPrime);
    net.SGD(training_data, test_data, params);

    // TODO : watch this https://www.youtube.com/watch?v=m7E9piHcfr4 to make this faster
    auto [correctTest, durationTest] = net.evaluate(test_data, test_data_size);
    auto [correctTrain, durationTrain] = net.evaluate(training_data, training_data_size);

    std::cerr << "accuracy in training data: " << (float) correctTrain / params.training_data_size << "\n";
    std::cerr << "general accuracy: " << (float) correctTest / params.test_data_size << "\n";

    // FIND-TAG-OUTPUT
    // cout << evtst.second << "\n";

    // FIND-TAG-STORING
    cerr << "Where should the network be stored? "; string filename; cin >> filename;
    // string filename = argv[1];
    net.save(filename);

    clear_data(test_data);
    clear_data(training_data);
    net.clear();
    delete[] layers;

    // FIND-TAG-MARK-END
    // cerr << "DONE";
}
