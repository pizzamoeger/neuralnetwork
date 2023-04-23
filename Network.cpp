#include "includes.h"

void Network::init(vector<int> &sizes, const function<double(double)> activationFunct,
                   const function<double(double)> activationFunctPrime,
                   const function<double(double, double)> costFunctPrime) {
    this->activationFunct = activationFunct;
    this->activationFunctPrime = activationFunctPrime;
    this->costFunctPrime = costFunctPrime;
    this->sizes = sizes;
    this->L = sizes.size();

    // initialize layers
    this->layers = vector<fully_connected_layer>(L);
    for (int i = 1; i < L; i++)
        layers[i].init(sizes[i - 1], sizes[i], activationFunct, activationFunctPrime, costFunctPrime);
}

pair<vector<vector<double>>, vector<vector<double>>> Network::feedforward(vector<double> &a) {
    vector<vector<double>> activations(L);
    vector<vector<double>> z(L);
    activations[0] = a;
    z[0] = a;
    for (int i = 0; i < sizes[0]; i++) z[0][i] = 1;

    for (int i = 1; i < L; i++) {
        z[i] = layers[i].feedforward(activations[i-1]);
        activations[i] = z[i];
        for (int j = 0; j < sizes[i]; j++) {
            activations[i][j] = activationFunct(z[i][j]);
        }
    }
    return {activations, z};
}

void Network::SGD(vector<pair<vector<double>, vector<double>>> training_data, vector<pair<vector<double>, vector<double>>> test_data, hyperparams params) {

    for (int i = 0; i < params.epochs; i++) {
        // reduce learning rate
        params.learning_rate *= 0.98;

        // time the epoch
        auto start = chrono::high_resolution_clock::now();

        cerr << i << " ";

        // obtain a time-based seed
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        shuffle(training_data.begin(), training_data.end(), default_random_engine(seed));

        // create mini batches and update them
        vector<pair<vector<double>, vector<double>>> mini_batch(params.mini_batch_size);
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
            vector<double> output = feedforward(test_data[k].first).first[L - 1];
            int max = 0;
            for (int j = 0; j < output.size(); j++) {
                if (output[j] > output[max]) max = j;
            }
            if (test_data[k].second[max] == 1) correct++;
        }
        end = chrono::high_resolution_clock::now();
        cerr << "Accuracy: " << (double) correct / test_data.size() << ", trained in " << duration.count()
             << "ms, evaluated in " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms\n";
    }
}

void Network::update_mini_batch(vector<pair<vector<double>, vector<double>>> &mini_batch, hyperparams params) {
    vector<vector<double>> updateBV(L);
    vector<vector<vector<double>>> updateWV(L);

    for (int i = 1; i < L; i++) updateBV[i] = vector<double>(sizes[i], 0);
    for (int i = 1; i < L; i++) updateWV[i] = vector<vector<double>>(sizes[i], vector<double>(sizes[i - 1], 0));

    for (auto [in, out]: mini_batch) backprop(in, out);

    // update velocities
    for (int i = 1; i < L; i++) layers[i].update(params);
}

void Network::backprop(vector<double> &in, vector<double> &out) {
    vector<vector<double>> updateB(L);
    vector<vector<vector<double>>> updateW(L);
    for (int i = 1; i < L; i++) updateB[i] = vector<double>(sizes[i], 0);
    for (int i = 1; i < L; i++) updateW[i] = vector<vector<double>>(sizes[i], vector<double>(sizes[i - 1], 0));

    // feedfoward
    auto [activations, z] = feedforward(in);

    // backpropagate
    vector<double> delta = vector<double>(sizes[L - 1], 0);
    for (int i = 0; i < sizes[L - 1]; i++) delta[i] = costFunctPrime(activations[L - 1][i], out[i]);

    for (int l = L - 1; l > 0; l--) layers[l].backprop(delta, activations[l-1], z[l-1]);
}

void Network::save(string filename) {
    ofstream file(filename);

    file << L << "\n";

    // sizes
    for (int i = 0; i < L; i++) file << sizes[i] << " ";
    file << "\n";

    // biases
    for (int i = 1; i < L; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            file << layers[i].biases[j] << " ";
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
        layers[i].biases = vector<double>(sizes[i]);
        for (int j = 0; j < sizes[i]; j++) {
            file >> layers[i].biases[j];
        }
    }

    // weights
    for (int i = 1; i < L; i++) {
        layers[i].weights = vector<vector<double>>(sizes[i]);
        for (int j = 0; j < sizes[i]; j++) {
            layers[i].weights[j] = vector<double>(sizes[i - 1]);
            for (int k = 0; k < sizes[i - 1]; k++) {
                file >> layers[i].weights[j][k];
            }
            char c;
            file >> c;
        }
    }

    file.close();
}