#include "includes.h"

void Network::init(layer_data* layers, int L, hyperparams params) {

    this->L = L;
    this->params = params;
    cudaMalloc((void**) &dev_params, sizeof(hyperparams));
    cudaMemcpy(dev_params, &params, sizeof(hyperparams), cudaMemcpyHostToDevice);
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

pair<float*, float*> Network::feedforward(float* a/*, float* dev_activations, float* dev_derivatives_z*/) {
    float* dev_activations;
    float* dev_derivatives_z;
    float* dev_z;
    int elems = layers[L-1]->data.elems+OUTPUT_NEURONS;

    cudaMalloc((void**) &dev_activations, elems*sizeof(float));
    cudaMalloc((void**) &dev_z, elems*sizeof(float));
    cudaMalloc((void**) &dev_derivatives_z, elems*sizeof(float));

    cudaMemcpy(dev_activations, a, 28*28*sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = 1; l < L; l++) {
        layers[l]->feedforward(dev_activations, dev_derivatives_z, dev_z);
    }

    cudaFree(dev_z);
    return {dev_activations, dev_derivatives_z};
}

pair<int,int> Network::evaluate(vector<pair<float*,float*>> test_data, int test_data_size) {
    auto start = chrono::high_resolution_clock::now();
    int correct = 0;

    int elems = layers[L-1]->data.elems+OUTPUT_NEURONS;
    /*float* activations;
    float* derivatives_z;
    float* dev_z;

    cudaMalloc((void**) &activations, elems*sizeof(float));
    cudaMalloc((void**) &dev_z, elems*sizeof(float));
    cudaMalloc((void**) &derivatives_z, elems*sizeof(float));*/

    for (int k = 0; k < (int) test_data_size; k++) {
        auto startFF = chrono::high_resolution_clock::now();
        auto ret = feedforward(test_data[k].first);
        auto activations = ret.first;
        auto dz = ret.second;
        auto endFF = chrono::high_resolution_clock::now();
        auto durFF = chrono::duration_cast<chrono::microseconds>(endFF - startFF).count();
        //cerr << durFF << "\n";

        int* dev_id_max;
        cudaMalloc((void**) &dev_id_max, sizeof(int));
        find_max<<<1,1>>> (&activations[layers[L-1]->data.elems], dev_id_max, &layers[L-1]->dev_data->n_out.x);

        // TODO: this can be done on device
        int id_max;
        cudaMemcpy(&id_max, dev_id_max, sizeof(int), cudaMemcpyDeviceToHost);
        float res;
        cudaMemcpy(&res, &test_data[k].second[id_max], sizeof(float), cudaMemcpyDeviceToHost);
        if (res == 1.0) correct++;
        // until here; copy memcpy only once: at the end for accuracy

        cudaFree(activations);
        cudaFree(dz);
    }
    auto end = chrono::high_resolution_clock::now();
    return {correct, chrono::duration_cast<chrono::milliseconds>(end - start).count()};
}

void Network::SGD(vector<pair<float*,float*>> training_data, vector<pair<float*,float*>> test_data) {

    auto ev = evaluate(test_data, params.test_data_size);
    auto correct = ev.first;
    auto durationEvaluate = ev.second;
    cerr << "0 Accuracy: " << (float) correct / params.test_data_size << " evaluated in " << durationEvaluate << "ms\n";

    for (int i = 0; i < params.epochs; i++) {
        // time the epoch
        auto start = chrono::high_resolution_clock::now();

        cerr << i+1 << " ";

        // obtain a time-based seed
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        shuffle(training_data.begin(), training_data.end(), default_random_engine(seed));

        // create mini batches and update them
        vector<pair<float*,float*>> mini_batch (params.mini_batch_size, {nullptr, nullptr});
        for (int j = 0; j < params.training_data_size / params.mini_batch_size; j++) {
            for (int k = 0; k < params.mini_batch_size; k++) {
                mini_batch[k].first = training_data[j * params.mini_batch_size + k].first;
                mini_batch[k].second = training_data[j * params.mini_batch_size + k].second;
            }
            update_mini_batch(mini_batch);
        }

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
            //cudaFree(dev_params);
            //cudaMalloc((void**) &dev_params, sizeof(hyperparams));
            cudaMemcpy(dev_params, &params, sizeof(hyperparams), cudaMemcpyHostToDevice);
        }
    }
}

void Network::update_mini_batch(vector<pair<float*,float*>> mini_batch) {

    for (int num = 0; num < params.mini_batch_size; num++) {
        backprop(mini_batch[num].first, mini_batch[num].second);
    }

    // update velocities
    for (int i = 1; i < L; i++) layers[i]->update(dev_params);
}

void Network::backprop(float* in, float* out) {
    // feedfoward

    pair<float*, float*> ret = feedforward(in);
    float* activations = ret.first;
    float* derivatives_z = ret.second;

    int elems = 0;
    for (int l = 0; l < L-1; l++) elems += layers[l]->data.n_out.x*layers[l]->data.n_out.y*layers[l]->data.n_out.feature_maps;
    int* dev_elems;
    cudaMalloc((void**) &dev_elems, sizeof(int));
    cudaMemcpy(dev_elems, &elems, sizeof(int), cudaMemcpyHostToDevice);

    // backpropagate
    float* delta;
    cudaMalloc((void**) &delta, OUTPUT_NEURONS*sizeof(float));
    set_to<<<OUTPUT_NEURONS, 1>>> (delta, 0);
    cudaDeviceSynchronize();

    set_delta<<<OUTPUT_NEURONS,1>>> (delta, &activations[elems], out, &dev_params->cost);

    for (int l = L - 1; l >= 1; l--) {
        layers[l]->backprop(delta, activations, derivatives_z, dev_elems);
        elems -= layers[l-1]->data.n_out.x*layers[l-1]->data.n_out.y*layers[l-1]->data.n_out.feature_maps;
        cudaMemcpy(dev_elems, &elems, sizeof(int), cudaMemcpyHostToDevice);
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

    cudaFree(dev_params);
    delete[] layers;
}