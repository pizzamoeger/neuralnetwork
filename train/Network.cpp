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
                new_layer = make_unique<convolutional_layer>();
                break;
            case LAYER_NUM_MAX_POOLING:
                new_layer = make_unique<max_pooling_layer>();
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

pair<float**, float**> Network::feedforward(input_type &a) {

    float** activations = new float* [L];
    float** derivatives_z = new float* [L];

    for (int l = 1; l < L; l++) {
        activations[l] = new float [layers[l]->data.n_out.x*layers[l]->data.n_out.y*layers[l]->data.n_out.feature_maps];
        derivatives_z[l] = new float [layers[l]->data.n_out.x*layers[l]->data.n_out.y*layers[l]->data.n_out.feature_maps];
        for (int i = 0; i < layers[l]->data.n_out.x*layers[l]->data.n_out.y*layers[l]->data.n_out.feature_maps; ++i) {
            activations[l][i] = 0;
            derivatives_z[l][i] = 0;
        }
    }

    activations[0] = a;
    derivatives_z[0] = a;

    for (int i = 1; i < L; i++) {
        layers[i]->feedforward(activations[i-1], derivatives_z[i-1], activations[i], derivatives_z[i]);
    }

    return {activations, derivatives_z};
}

pair<int,int> Network::evaluate(data_point* test_data, int test_data_size) {
    auto start = chrono::high_resolution_clock::now();
    int correct = 0;
    for (int k = 0; k < (int) test_data_size; k++) {
        auto [activations, derivatives_z] = feedforward(test_data[k].first);
        float* output = activations[L-1];
        int max = 0;
        for (int j = 0; j < 10; j++) {
            if (output[j] > output[max]) max = j;
        }
        if (test_data[k].second[max] == 1.0) correct++;
        for (int l = L - 1; l > 0; l--) {
            delete[] activations[l];
            delete[] derivatives_z[l];
        }
        delete[] activations;
        delete[] derivatives_z;
    }
    auto end = chrono::high_resolution_clock::now();
    return {correct, chrono::duration_cast<chrono::milliseconds>(end - start).count()};
}

void Network::SGD(data_point* training_data, data_point* test_data, hyperparams params) {

    auto [correct, durationEvaluate] = evaluate(test_data, params.test_data_size);
    cerr << "0 Accuracy: " << (float) correct / params.test_data_size << " evaluated in " << durationEvaluate << "ms\n";

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
                    mini_batch[k].first[p] =  training_data[j * params.mini_batch_size + k].first[p];
                }
                for (int c = 0; c < 10; c++) {
                    mini_batch[k].second[c] = training_data[j * params.mini_batch_size + k].second[c];
                }
            }
            update_mini_batch(mini_batch, params);
        }
        delete[] mini_batch;

        // end the timer
        auto end = chrono::high_resolution_clock::now();
        auto durationTrain = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        // evaluate the network
        auto [correct, durationEvaluate] = evaluate(test_data, params.test_data_size);

        cerr << "Accuracy: " << (float) correct / params.test_data_size << ", trained in " << durationTrain << "ms, evaluated in " << durationEvaluate << "ms\n";

        // reduce learning rate
	    if (i < 100) {
            params.fully_connected_biases_learning_rate -= params.fcBRed;
            params.fully_connected_weights_learning_rate -= params.fcWRed;
            params.convolutional_biases_learning_rate -= params.convBRed;
            params.convolutional_weights_learning_rate -= params.convWRed;
        }
    }
}

void Network::update_mini_batch(data_point* mini_batch, hyperparams params) {

    for (int num = 0; num < params.mini_batch_size; num++) {
        backprop(mini_batch[num].first, mini_batch[num].second);
    }

    // update velocities
    for (int i = 1; i < L; i++) layers[i]->update(params);
}

void Network::backprop(input_type &in, output_type &out) {
    // feedfoward
    auto [activations, derivatives_z] = feedforward(in);

    // backpropagate
    vector<float> delta = vector<float> (layers[L-1]->data.n_out.x*layers[L-1]->data.n_out.y*layers[L-1]->data.n_out.feature_maps, 0);
    for (int i = 0; i < layers[L-1]->data.n_out.x*layers[L-1]->data.n_out.y*layers[L-1]->data.n_out.feature_maps; i++) delta[i] = costFunctPrime(activations[L - 1][i], out[i]);

    for (int l = L - 1; l >= 1; l--) layers[l]->backprop(delta, activations[l-1], derivatives_z[l]);

    // clean
    for (int l = L - 1; l > 0; l--) {
        delete[] activations[l];
        delete[] derivatives_z[l];
    }
    delete[] activations;
    delete[] derivatives_z;
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
/*
void Network::load(string filename) {
    ifstream file(filename);

    file >> L;

    // sizes
    sizes = vector<int>(L);
    for (int i = 0; i < L; i++) file >> sizes[i];

    layers = vector<fully_connected_layer>(L);
    for (int i = 1; i < L; i++) layers[i].init(sizes[i - 1], sizes[i], activationFunct, activationFunctPrime, costFunctPrime);

    // biases
    for (int i = 1; i < L; i++) {
        layers[i].biases = vector<float>(sizes[i]);
        for (int j = 0; j < sizes[i]; j++) {
            file >> layers[i].biases[j];
        }
    }

    // weights
    for (int i = 1; i < L; i++) {
        layers[i].weights = vector<vector<float>>(sizes[i]);
        for (int j = 0; j < sizes[i]; j++) {
            layers[i].weights[j] = vector<float>(sizes[i - 1]);
            for (int k = 0; k < sizes[i - 1]; k++) {
                file >> layers[i].weights[j][k];
            }
            char c;
            file >> c;
        }
    }

    file.close();
}*/
