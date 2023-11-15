#include "includes.h"
int* zero_pointer;
float* f_zero_pointer;

int main(int argc, char** argv) {

    // zeros to use on GPU
    cudaGetSymbolAddress((void**) &zero_pointer, zero);
    cudaGetSymbolAddress((void**) &f_zero_pointer, zero);

    // randomness
    srand(time(NULL));

    Network net;

    // design the layers
    layer_data input;
    input.type = LAYER_NUM_INPUT;
    input.n_out = {28, 28, 1};
    //input.n_out = {NEURONS, 1, 1};

    layer_data convolutional;
    convolutional.type = LAYER_NUM_CONVOLUTIONAL;
    convolutional.stride_length = 1;
    convolutional.receptive_field_length = 7;
    convolutional.activation_function = RELU;
    convolutional.n_out = {-1,-1, 2};

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
    fully_connected2.n_out = {50, 1, 1};

    layer_data outt;
    outt.type = LAYER_NUM_FULLY_CONNECTED;
    outt.activation_function = RELU;
    outt.last_layer = true;
    outt.n_out = {OUTPUT_NEURONS, 1, 1};
    //outt.n_out = {NEURONS, 1, 1};

    // design the network
    // FIND-TAG-LAYERS
    int L = 4;
    layer_data* layers = new layer_data[L];
    layers[0] = input;
    layers[1] = convolutional;
    layers[2] = maxpool;
    layers[2] = fully_connected2;
    layers[2] = fully_connected2;
    layers[3] = outt;

    // load data
    auto tst = load_data("mnist_test_normalized.data");
    std::vector<std::pair<float*, float*>> test_data = tst.first;
    int test_data_size = tst.second;
    auto trn = load_data("mnist_train_normalized.data");
    std::vector<std::pair<float*, float*>>  training_data = trn.first;
    int training_data_size = trn.second;

    // get hyperparams
    auto params = get_params();

    if (argc == 7) {
        // read hyperparams from commandline arguments
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
    std::cerr << "epochs: "; std::cin >> params.epochs;
    // params.epochs = 150;
    // params.epochs = 0;

    // train network
    net.init(layers, L, params);
    net.SGD(training_data, test_data);

    // get accuracy of network
    auto evtst = net.evaluate(test_data, test_data_size);
    int correct_test = evtst.first;
    auto evtrn = net.evaluate(training_data, training_data_size);
    int correct_train = evtrn.first;

    std::cerr << "accuracy in training data: " << (float) correct_train / params.training_data_size << "\n";
    std::cerr << "general accuracy: " << (float) correct_test / params.test_data_size << "\n";

    // FIND-TAG-OUTPUT
    // cout << evtst.second << "\n";

    // save network
    // FIND-TAG-STORING
    std::cerr << "Where should the network be stored? "; std::string filename; std::cin >> filename;
    // string filename = argv[1];
    net.save(filename);

    // free up memory
    clear_data(test_data);
    clear_data(training_data);
    net.clear();
    delete[] layers;

    cudaDeviceReset();
}
