#include "includes.h"

void Network::init(vector<layer_data> & layers, function<float(float, float)> costFunctPrime) {

    this->L = layers.size();
    this->costFunctPrime = costFunctPrime;

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
        this->layers.push_back(move(new_layer));
    }
}

pair<vector<vector<float>>, vector<vector<float>>> Network::feedforward(vector<float> &a) {
    vector<vector<float>> activations(L, vector<float> (1));
    vector<vector<float>> derivatives_z(L, vector<float> (1));

    activations[0] = a;
    derivatives_z[0] = a;

    for (int i = 1; i < L; i++) {
        activations[i] = activations[i-1];
        derivatives_z[i] = derivatives_z[i-1];
        layers[i]->feedforward(activations[i], derivatives_z[i]);
    }

    return {activations, derivatives_z};
}

pair<int,int> Network::evaluate(vector<pair<vector<float>, vector<float>>> test_data, hyperparams params) {
    auto start = chrono::high_resolution_clock::now();
    int correct = 0;
    for (int k = 0; k < (int) test_data.size(); k++) {
        vector<float> output = feedforward(test_data[k].first).first[L - 1];
        int max = 0;
        for (int j = 0; j < (int)output.size(); j++) {
            if (output[j] > output[max]) max = j;
        }
        if (test_data[k].second[max] == 1) correct++;
    }
    auto end = chrono::high_resolution_clock::now();
    return {correct, chrono::duration_cast<chrono::milliseconds>(end - start).count()};
}

void Network::SGD(vector<pair<vector<float>, vector<float>>> training_data, vector<pair<vector<float>, vector<float>>> test_data, hyperparams params) {
    auto [correct, durationEvaluate] = evaluate(test_data, params);
    cerr << "0 Accuracy: " << (float) correct / params.test_data_size << " evaluated in " << durationEvaluate << "ms\n";

    for (int i = 0; i < params.epochs; i++) {
        // time the epoch
        auto start = chrono::high_resolution_clock::now();

        cerr << i+1 << " ";

        // obtain a time-based seed
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        shuffle(training_data.begin(), training_data.end(), default_random_engine(seed));

        // create mini batches and update them
        vector<pair<vector<float>, vector<float>>> mini_batch(params.mini_batch_size);
        for (int j = 0; j < params.training_data_size / params.mini_batch_size; j++) {
            for (int k = 0; k < params.mini_batch_size && j * params.mini_batch_size + k < params.training_data_size; k++) {
                mini_batch[k] = training_data[j * params.mini_batch_size + k];
            }
            update_mini_batch(mini_batch, params);
        }

        // end the timer
        auto end = chrono::high_resolution_clock::now();
        auto durationTrain = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        // evaluate the network
        auto [correct, durationEvaluate] = evaluate(test_data, params);

        cerr << "Accuracy: " << (float) correct / params.test_data_size << ", trained in " << durationTrain << "ms, evaluated in " << durationEvaluate << "ms\n";

        // reduce learning rate
        params.fully_connected_biases_learning_rate *= 0.97;
        params.fully_connected_weights_learning_rate *= 0.97;
        params.convolutional_weights_learning_rate *= 0.97;
        params.convolutional_biases_learning_rate *= 0.97;
    }
}

void Network::update_mini_batch(vector<pair<vector<float>, vector<float>>> &mini_batch, hyperparams params) {

    for (auto [in, out]: mini_batch) {
        backprop(in, out);
    }

    // update velocities
    for (int i = 1; i < L; i++)
        layers[i]->update(params);
}

void Network::backprop(vector<float> &in, vector<float> &out) {
    // feedfoward
    auto [activations, derivatives_z] = feedforward(in);

    // backpropagate
    vector<float> delta = vector<float> (activations[L-1].size(), 0);
    for (int i = 0; i < (int)activations[L-1].size(); i++) delta[i] = costFunctPrime(activations[L - 1][i], out[i]);

    for (int l = L - 1; l >= 1; l--) layers[l]->backprop(delta, activations[l-1], derivatives_z[l]);
}

/* uiuiui da isch ganz anders ez
void Network::save(string filename) {
    ofstream file(filename);

    file << L << "\n";

    // sizes
    for (int i = 0; i < L; i++) file << layers[i]->n_out.x*layers[i]->n_out.y << " ";
    file << "\n";

    // biases
    for (int i = 1; i < L; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            file << layers[i]->biases[j] << " ";
        }
        file << "\n";
    }

    // weights
    for (int i = 1; i < L; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            for (int k = 0; k < sizes[i - 1]; k++) {
                file << layers[i].weights[j][k] << " ";
            }
            file << "^";
        }
        file << "\n";
    }

    file.close();
}

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