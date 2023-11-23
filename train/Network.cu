#include "includes.h"

void Network::init(layer_data* layers, int L, hyperparams params) {

    this->L = L;
    this->params = params;
    cudaMalloc((void**) &dev_params, sizeof(hyperparams));
    cudaMemcpy(dev_params, &params, sizeof(hyperparams), cudaMemcpyHostToDevice);
    this->layers = new std::unique_ptr<layer>[L];

    // initialize layers
    for (int l = 0; l < L; l++) {
        std::unique_ptr<layer> new_layer = nullptr;
        switch (layers[l].type) {
            case LAYER_NUM_INPUT:
                new_layer = std::make_unique<input_layer>();
                break;
            case LAYER_NUM_CONVOLUTIONAL:
                new_layer = std::make_unique<convolutional_layer>();
                break;
            case LAYER_NUM_MAX_POOLING:
                //new_layer = make_unique<max_pooling_layer>();
                break;
            case LAYER_NUM_FULLY_CONNECTED:
                new_layer = std::make_unique<fully_connected_layer>();
                break;
        }
        layer_data previous_data;
        float* new_delta;
        if (l > 0) {
            previous_data = this->layers[l - 1]->data;
            new_delta = this->layers[l-1]->delta;
        }
        new_layer->init(layers[l], previous_data, new_delta);
        this->layers[l] = move(new_layer);
    }

    // initialize activations and dz
    int elems = this->layers[L-1]->data.elems+OUTPUT_NEURONS;

    cudaMalloc((void**) &activations, elems*sizeof(float));
    cudaMalloc((void**) &derivatives_z, elems*sizeof(float));
}

void Network::feedforward(float* a, float* dev_activations, float* dev_derivatives_z) {

    cudaMemcpy(dev_activations, a, INPUT_NEURONS*sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = 1; l < L; l++) {
        layers[l]->feedforward(dev_activations, dev_derivatives_z);
    }
}

std::pair<int,int> Network::evaluate(std::vector<std::pair<float*,float*>> test_data, int test_data_size) {
    auto start = std::chrono::high_resolution_clock::now();

    int* dev_correct;
    cudaMalloc((void**) &dev_correct, sizeof(int));
    cudaMemcpy(dev_correct, zero_pointer, sizeof(int), cudaMemcpyDeviceToDevice);

    for (int k = 0; k < (int) test_data_size; k++) {
        feedforward(test_data[k].first, activations, derivatives_z);
        cudaDeviceSynchronize();
        // TODO: this could be made faster but not really necessary, as it's only 10 -> is already fast
        eval<<<1,1>>>(test_data[k].second, &activations[layers[L-1]->data.elems], dev_correct, &layers[L-1]->dev_data->n_out.x);
    }
    cudaDeviceSynchronize();

    int correct;
    cudaMemcpy(&correct, dev_correct, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_correct);
    auto end = std::chrono::high_resolution_clock::now();
    return {correct, std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()};
}

void Network::SGD(std::vector<std::pair<float*,float*>> training_data, std::vector<std::pair<float*,float*>> test_data) {

    auto ev = evaluate(test_data, params.test_data_size);
    auto correct = ev.first;
    auto durationEvaluate = ev.second;
    std::cerr << "0 Accuracy: " << (float) correct / params.test_data_size << " evaluated in " << durationEvaluate << "ms\n";

    for (int i = 0; i < params.epochs; i++) {
        // time the epoch
        auto start = std::chrono::high_resolution_clock::now();

        std::cerr << i+1 << " ";

        // obtain a time-based seed
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(training_data.begin(), training_data.end(), std::default_random_engine(seed));

        // create mini batches and update them
        std::vector<std::pair<float*,float*>> mini_batch (params.mini_batch_size, {nullptr, nullptr});
        for (int j = 0; j < params.training_data_size / params.mini_batch_size; j++) {
            for (int k = 0; k < params.mini_batch_size; k++) {
                mini_batch[k].first = training_data[j * params.mini_batch_size + k].first;
                mini_batch[k].second = training_data[j * params.mini_batch_size + k].second;
            }
            update_mini_batch(mini_batch);
        }

        // end the timer
        auto end = std::chrono::high_resolution_clock::now();
        auto durationTrain = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // evaluate the network
        ev = evaluate(test_data, params.test_data_size);
        correct = ev.first;
        durationEvaluate = ev.second;

        std::cerr << "Accuracy: " << (float) correct / params.test_data_size << ", trained in " << durationTrain << "ms, evaluated in " << durationEvaluate << "ms\n";

        // reduce learning rate
	    if (i < 100) {
            params.fully_connected_biases_learning_rate -= params.fcBRed;
            params.fully_connected_weights_learning_rate -= params.fcWRed;
            params.convolutional_biases_learning_rate -= params.convBRed;
            params.convolutional_weights_learning_rate -= params.convWRed;
            cudaMemcpy(dev_params, &params, sizeof(hyperparams), cudaMemcpyHostToDevice);
        }
    }
}

void Network::update_mini_batch(std::vector<std::pair<float*,float*>> mini_batch) {

    for (int num = 0; num < params.mini_batch_size; num++) {
        backprop(mini_batch[num].first, mini_batch[num].second);
    }

    // update velocities
    for (int i = 1; i < L; i++) layers[i]->update(dev_params);
}

void Network::backprop(float* in, float* out) {
    // feedfoward
    feedforward(in, activations, derivatives_z);

    // backpropagate
    set_delta<<<OUTPUT_NEURONS,1>>> (layers[L-1]->delta, &activations[layers[L-1]->data.elems], out, &dev_params->cost);

    for (int l = L - 1; l >= 1; l--) {
        layers[l]->backprop(activations, derivatives_z);
    }
}

void Network::save(std::string filename) {
    std::ofstream file(filename);
    file << L << "\n";
    file.close();

    for (int l = 0; l < L; l++) layers[l]->save(filename);
}

void Network::clear() {
    for (int l = 0; l < L; l++) layers[l]->clear();

    cudaFree(dev_params);
    cudaFree(activations);
    cudaFree(derivatives_z);
    delete[] layers;
}