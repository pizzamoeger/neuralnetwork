#include "includes.h"

void Network::init(layer_data* layers, int L, function<float(float, float)> costFunctPrime) {

    this->L = L;
    this->costFunctPrime = costFunctPrime;
    this->layers = new unique_ptr<layer>[L];

    // initialize layers
    for (int l = 0; l < L; l++) {
        unique_ptr<layer> new_layer = nullptr;
        switch (layers[l].type) {
            case LAYER_NUM_INPUT:
                new_layer = make_unique<input_layer>();
                break;
            case LAYER_NUM_CONVOLUTIONAL:
                //new_layer = make_unique<convolutional_layer>();
                break;
            case LAYER_NUM_MAX_POOLING:
                //new_layer = make_unique<max_pooling_layer>();
                break;
            case LAYER_NUM_FULLY_CONNECTED:
                new_layer = make_unique<fully_connected_layer>();
                break;
        }
        layer_data previous_data;
        if (l > 0) previous_data = this->layers[l - 1]->data;
        new_layer->init(layers[l], previous_data);
        this->layers[l] = move(new_layer);
    }
}

pair<float*, float*> Network::feedforward(input_type &a) {
    float* dev_activations;
    float* dev_derivatives_z;
    float* dev_z;
    int elems = 0;
    for (int l = 0; l < L; l++) elems += layers[l]->data.n_out.x*layers[l]->data.n_out.y*layers[l]->data.n_out.feature_maps;

    cudaMalloc((void**) &dev_activations, elems*sizeof(float));
    cudaMalloc((void**) &dev_z, elems*sizeof(float));
    cudaMalloc((void**) &dev_derivatives_z, elems*sizeof(float));
    set_to<<<elems,1>>>(dev_activations, 0);
    set_to<<<elems,1>>>(dev_z, 0);
    set_to<<<elems,1>>>(dev_derivatives_z, 0);

    cudaMemcpy(dev_activations, a, 28*28*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_derivatives_z, a, 28*28*sizeof(float), cudaMemcpyHostToDevice);

    elems = layers[0]->data.n_out.x*layers[0]->data.n_out.y*layers[0]->data.n_out.feature_maps;;
    int* dev_elems;
    cudaMalloc((void**) &dev_elems, sizeof(int));
    cudaMemcpy(dev_elems, &elems, sizeof(int), cudaMemcpyHostToDevice);

    for (int l = 1; l < L; l++) {
        layers[l]->feedforward(dev_activations, dev_derivatives_z, dev_z, dev_elems);
        elems += layers[l]->data.n_out.x*layers[l]->data.n_out.y*layers[l]->data.n_out.feature_maps;
        cudaMemcpy(dev_elems, &elems, sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaFree(dev_z);
    cudaFree(dev_elems);
    return {dev_activations, dev_derivatives_z};
}

pair<int,int> Network::evaluate(data_point* test_data, int test_data_size) {
    auto start = chrono::high_resolution_clock::now();
    int correct = 0;
    int elems = 0;
    for (int l = 0; l < L-1; l++) elems += layers[l]->data.n_out.x*layers[l]->data.n_out.y*layers[l]->data.n_out.feature_maps;

    for (int k = 0; k < (int) test_data_size; k++) {
        auto ret = feedforward(test_data[k].first);
        auto activations = ret.first;
        auto derivatives_z = ret.second;
        output_type output;
        cudaMemcpy(output, &activations[elems], sizeof(output_type), cudaMemcpyDeviceToHost);

        int max = 0;
        for (int j = 0; j < 10; j++) {
            if (output[j] > output[max]) max = j;
        }
        if (test_data[k].second[max] == 1.0) correct++;

        cudaFree(activations);
        cudaFree(derivatives_z);
    }
    auto end = chrono::high_resolution_clock::now();
    return {correct, chrono::duration_cast<chrono::milliseconds>(end - start).count()};
}

void Network::SGD(data_point* training_data, data_point* test_data, hyperparams params) {

    auto ev = evaluate(test_data, params.test_data_size);
    auto correct = ev.first;
    auto durationEvaluate = ev.second;
    cerr << "0 Accuracy: " << (float) correct / params.test_data_size << " evaluated in " << durationEvaluate << "ms\n";

    hyperparams* dev_params;
    cudaMalloc((void**) &dev_params, sizeof(hyperparams));
    cudaMemcpy(dev_params, &params, sizeof(hyperparams), cudaMemcpyHostToDevice);

    for (int i = 0; i < params.epochs; i++) {
        // time the epoch
        auto start = chrono::high_resolution_clock::now();

        cerr << i+1 << " ";

        // obtain a time-based seed
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        shuffle(training_data, training_data+params.training_data_size, default_random_engine(seed));

        // create mini batches and update them
        auto* mini_batch = new data_point [params.mini_batch_size];
        for (int j = 0; j < params.training_data_size / params.mini_batch_size; j++) {
            for (int k = 0; k < params.mini_batch_size; k++) {
                for (int p = 0; p < 784; p++) {
                    mini_batch[k].first[p] = training_data[j * params.mini_batch_size + k].first[p];
                }
                for (int c = 0; c < 10; c++) {
                    mini_batch[k].second[c] = training_data[j * params.mini_batch_size + k].second[c];
                }
            }
            update_mini_batch(mini_batch, params, dev_params);
        }
        delete[] mini_batch;

        // end the timer
        auto end = chrono::high_resolution_clock::now();
        auto durationTrain = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        // evaluate the network
        ev = evaluate(test_data, params.test_data_size);
        correct = ev.first;
        durationEvaluate = ev.second;

        cerr << "Accuracy: " << (float) correct / params.test_data_size << ", trained in " << durationTrain << "ms, evaluated in " << durationEvaluate << "ms\n";

        // reduce learning rate
	    if (i < 100) {
            params.fully_connected_biases_learning_rate -= params.fcBRed;
            params.fully_connected_weights_learning_rate -= params.fcWRed;
            params.convolutional_biases_learning_rate -= params.convBRed;
            params.convolutional_weights_learning_rate -= params.convWRed;
            cudaMemcpy(dev_params, &params, sizeof(hyperparams), cudaMemcpyHostToDevice);
        }
    }
    cudaFree(dev_params);
}

void Network::update_mini_batch(data_point* mini_batch, hyperparams params, hyperparams* dev_params) {

    for (int num = 0; num < params.mini_batch_size; num++) {
        backprop(mini_batch[num].first, mini_batch[num].second);
    }

    // update velocities
    for (int i = 1; i < L; i++) layers[i]->update(dev_params);
}

void Network::backprop(input_type &in, output_type &out) {
    // feedfoward
    pair<float*, float*> ret = feedforward(in);
    float* activations = ret.first;
    float* derivatives_z = ret.second;

    int elems = 0;
    for (int l = 0; l < L; l++) elems += layers[l]->data.n_out.x*layers[l]->data.n_out.y*layers[l]->data.n_out.feature_maps;
    int* dev_elems;
    cudaMalloc((void**) &dev_elems, sizeof(int));

    // backpropagate
    float* delta;
    output_type host_delta;
    output_type host_activations;

    cudaMalloc((void**) &delta, 10*sizeof(float));
    cudaMemcpy(host_activations, &activations[elems-10], sizeof(output_type), cudaMemcpyDeviceToHost);

    for (int neuron = 0; neuron < 10; neuron++) host_delta[neuron] = costFunctPrime(host_activations[neuron], out[neuron]);
    cudaMemcpy(delta, host_delta, sizeof(output_type), cudaMemcpyHostToDevice);

    for (int l = L - 1; l >= 1; l--) {
        elems -= layers[l]->data.n_out.x*layers[l]->data.n_out.y*layers[l]->data.n_out.feature_maps;
        cudaMemcpy(dev_elems, &elems, sizeof(int), cudaMemcpyHostToDevice);
        layers[l]->backprop(delta, activations, derivatives_z, dev_elems);
    }

    // clean
    cudaFree(dev_elems);
    cudaFree(delta);
    cudaFree(activations);
    cudaFree(derivatives_z);
}

void Network::save(string filename) {
    ofstream file(filename);
    file << L << "\n";
    file.close();

    for (int l = 0; l < L; l++) layers[l]->save(filename);
}

void Network::clear() {
    for (int l = 0; l < L; l++) layers[l]->clear();

    delete[] layers;
}