#include "includes.h"

void Network::init(vector<layer_data> & layers, function<float(float, float)> costFunctPrime) {

    this->L = layers.size();
    this->costFunctPrime = costFunctPrime;

    // initialize layers
    for (int l = 0; l < L; l++) {
        unique_ptr<layer> new_layer = nullptr;
        switch (layers[l].type) {
            case 0:
                new_layer = make_unique<input_layer>();
                break;
            case 1:
                new_layer = make_unique<convolutional_layer>();
                break;
            case 2:
                new_layer = make_unique<max_pooling_layer>();
                break;
            case 3:
                new_layer = make_unique<flatten_layer>();
                break;
            case 4:
                new_layer = make_unique<fully_connected_layer>();
                break;
        }
        new_layer->init(layers[l]);
        this->layers.push_back(move(new_layer));
    }
}

pair<vector<vector<vector<vector<float>>>>, vector<vector<vector<vector<float>>>>> Network::feedforward(vector<vector<float>> &a) {
    vector<vector<vector<vector<float>>>> activations(L, vector<vector<vector<float>>> (1));
    vector<vector<vector<vector<float>>>> derivatives_z(L, vector<vector<vector<float>>> (1));

    activations[0][0] = a;
    derivatives_z[0][0] = a;

    for (int i = 1; i < L; i++) {
        activations[i] = activations[i-1];
        derivatives_z[i] = derivatives_z[i-1];
        layers[i]->feedforward(activations[i], derivatives_z[i]);
    }

    return {activations, derivatives_z};
}

void Network::SGD(vector<pair<vector<vector<float>>, vector<float>>> training_data, vector<pair<vector<vector<float>>, vector<float>>> test_data, hyperparams params) {

    for (int i = 0; i < params.epochs; i++) {
        // time the epoch
        auto start = chrono::high_resolution_clock::now();

        cerr << i << " ";

        // obtain a time-based seed
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        shuffle(training_data.begin(), training_data.end(), default_random_engine(seed));

        // create mini batches and update them
        vector<pair<vector<vector<float>>, vector<float>>> mini_batch(params.mini_batch_size);
        for (int j = 0; j < params.training_data_size / params.mini_batch_size; j++) {
            for (int k = 0; k < params.mini_batch_size && j * params.mini_batch_size + k < params.training_data_size; k++) {
                mini_batch[k] = training_data[j * params.mini_batch_size + k];
            }
            update_mini_batch(mini_batch, params);
        }

        // end the timer
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        // evaluate the network
        start = chrono::high_resolution_clock::now();
        int correct = 0;
        for (int k = 0; k < test_data.size(); k++) {
            vector<float> output = feedforward(test_data[k].first).first[L - 1][0][0];
            int max = 0;
            for (int j = 0; j < output.size(); j++) {
                if (output[j] > output[max]) max = j;
            }
            if (test_data[k].second[max] == 1) correct++;
        }
        end = chrono::high_resolution_clock::now();
        cerr << "Accuracy: " << (float) correct / test_data.size() << ", trained in " << duration.count()
             << "ms, evaluated in " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms\n";

        // reduce learning rate
        params.fully_connected_biases_learning_rate *= 0.97;
        params.fully_connected_weights_learning_rate *= 0.97;
        params.convolutional_weights_learning_rate *= 0.97;
        params.convolutional_biases_learning_rate *= 0.97;
    }
}

void Network::update_mini_batch(vector<pair<vector<vector<float>>, vector<float>>> &mini_batch, hyperparams params) {

    for (auto [in, out]: mini_batch) backprop(in, out);

    // update velocities
    for (int i = 1; i < L; i++)
        layers[i]->update(params);
}

void Network::backprop(vector<vector<float>> &in, vector<float> &out) {
    // feedfoward
    auto [activations, derivatives_z] = feedforward(in);

    // backpropagate
    vector<vector<vector<float>>> delta = vector<vector<vector<float>>> (1, vector<vector<float>> (1 ,vector<float>(activations[L-1][0][0].size(), 0)));
    for (int i = 0; i < activations[L-1][0][0].size(); i++) delta[0][0][i] = costFunctPrime(activations[L - 1][0][0][i], out[i]);

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