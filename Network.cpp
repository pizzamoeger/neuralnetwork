#include "includes.h"

default_random_engine generator;
normal_distribution<double> distribution(0.0, 1.0 / sqrt(784));

void Network::init (vector<int> & sizes, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime) {
    this->activationFunct = activationFunct;
    this->activationFunctPrime = activationFunctPrime;
    this->costFunctPrime = costFunctPrime;
    this->sizes = sizes;
    this->L = sizes.size();

    // initialize biases
    biases = vector<vector<double>> (L);
    biasesVelocity = vector<vector<double>> (L);
    for (int i = 1; i < L; i++) {
        biases[i] = vector<double> (sizes[i], distribution(generator));
        biasesVelocity[i] = vector<double> (sizes[i], 0);
    }

    // initialize weights
    weights = vector<vector<vector<double>>> (L);
    weightsVelocity = vector<vector<vector<double>>> (L);
    for (int i = 1; i < L; i++) {
        weights[i] = vector<vector<double>> (sizes[i]);
        weightsVelocity[i] = vector<vector<double>> (sizes[i]);
        for (int j = 0; j < sizes[i]; j++) {
            weights[i][j] = vector<double> (sizes[i-1], distribution(generator));
            weightsVelocity[i][j] = vector<double> (sizes[i-1], 0);
        }
    }
}

vector<double> Network::feedforward(vector<double> & a) {
    for (int i = 1; i < L; i++) {
        vector<double> newA (sizes[i], 0);
        for (int j = 0; j < sizes[i]; j++) {
            // get new activations
            for (int k = 0; k < sizes[i-1]; k++) newA[j] += weights[i][j][k]*a[k];
            newA[j] += biases[i][j];
            newA[j] = activationFunct(newA[j]);
        }
        // update a
        a = newA;
    }
    return a;
}

void Network::SGD(vector<pair<vector<double>,vector<double>>> training_data, int epochs, int mini_batch_size, double learning_rate, vector<pair<vector<double>, vector<double>>> test_data, double lambda, double momentum_coefficient) {
    int n = training_data.size();
    for (int i = 0; i < epochs; i++) {
        // reduce learning rate
        learning_rate *= 0.98;

        // time the epoch
        auto start = chrono::high_resolution_clock::now();

        cerr << i << " ";

        // obtain a time-based seed
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        shuffle(training_data.begin(), training_data.end(), default_random_engine(seed));

        // create mini batches and update them
        vector<pair<vector<double>, vector<double>>> mini_batch (mini_batch_size);
        for (int j = 0; j < n/mini_batch_size; j++) {
            for (int k = 0; k < mini_batch_size && j*mini_batch_size+k<n; k++) {
                mini_batch[k] = training_data[j*mini_batch_size+k];
            }
            update_mini_batch(mini_batch, learning_rate, lambda, n, momentum_coefficient);
        }

        // end the timer
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        // evaluate the network
        start = chrono::high_resolution_clock::now();
        int correct = 0;
        for (int k = 0; k < test_data.size(); k++) {
            vector<double> output = feedforward(test_data[k].first);
            int max = 0;
            for (int j = 0; j < output.size(); j++) {
                if (output[j] > output[max]) max = j;
            }
            if (test_data[k].second[max] == 1) correct++;
        }
        end = chrono::high_resolution_clock::now();
        cerr << "Accuracy: " << (double)correct / test_data.size() << ", trained in " << duration.count() << "ms, evaluated in " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms\n";
    }
}

void Network::update_mini_batch(vector<pair<vector<double>,vector<double>>> & mini_batch, double learning_rate, double lambda, int n, double momentum_coefficient) {
    vector<vector<double>> updateBV (L);
    vector<vector<vector<double>>> updateWV (L);

    for (int i = 1; i < L; i++) updateBV[i] = vector<double> (sizes[i], 0);
    for (int i = 1; i < L; i++) updateWV[i] = vector<vector<double>> (sizes[i], vector<double> (sizes[i-1], 0));

    for (auto [in, out] : mini_batch) {
        // get the errors
        auto ret = backprop(in, out);
        auto deltaB = ret.first;
        auto deltaW = ret.second;

        // add them to the current error
        for (int i = 1; i < L; i++) {
            for (int j = 0; j < sizes[i]; j++) {
                updateBV[i][j] += deltaB[i][j];
            }
        }
        for (int i = 1; i < L; i++) {
            for (int j = 0; j < sizes[i]; j++) {
                for (int k = 0; k < sizes[i-1]; k++) updateWV[i][j][k] += deltaW[i][j][k];
            }
        }
    }


    // update velocities
    for (int i = 1; i < L; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            biasesVelocity[i][j] = momentum_coefficient*biasesVelocity[i][j]-(learning_rate/mini_batch.size())*updateBV[i][j];
        }
    }

    for (int i = 1; i < L; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            for (int k = 0; k < sizes[i-1]; k++) {
                weightsVelocity[i][j][k] = momentum_coefficient*weightsVelocity[i][j][k]-(learning_rate/mini_batch.size())*updateWV[i][j][k];
            }
        }
    }

    // update weights and biases
    for (int i = 1; i < L; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            biases[i][j] = biases[i][j]+biasesVelocity[i][j];
        }
    }
    for (int i = 1; i < L; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            for (int k = 0; k < sizes[i-1]; k++) {
                weights[i][j][k] = (1-learning_rate*lambda/n)*weights[i][j][k]+weightsVelocity[i][j][k];
            }
        }
    }
}

pair<vector<vector<double>>, vector<vector<vector<double>>>> Network::backprop(vector<double> & in, vector<double> & out) {
    vector<vector<double>> updateB (L);
    vector<vector<vector<double>>> updateW (L);
    for (int i = 1; i < L; i++) updateB[i] = vector<double> (sizes[i], 0);
    for (int i = 1; i < L; i++) updateW[i] = vector<vector<double>> (sizes[i], vector<double> (sizes[i-1], 0));

    // feedfoward
    vector<vector<double>> z (L);
    vector<vector<double>> activations (L);
    activations[0] = in;

    for (int i = 1; i < L; i++) {
        activations[i].assign(sizes[i], 0);
        z[i].assign(sizes[i], 0);
        for (int j = 0; j < sizes[i]; j++) {
            // get new activations
            for (int k = 0; k < sizes[i-1]; k++) {
                z[i][j] += weights[i][j][k]*activations[i-1][k];
            }
            z[i][j] += biases[i][j];
            activations[i][j] = activationFunct(z[i][j]);
        }
    }

    // backpropagate
    vector<double> delta = vector<double> (sizes[L-1], 0);
    for (int i = 0; i < delta.size(); i++) delta[i] = costFunctPrime(activations[L-1][i], out[i]);

    updateB[L-1] = delta;
    for (int i = 0; i < delta.size(); i++) {
        for (int j = 0; j < activations[L-2].size(); j++) updateW[L-1][i][j] = delta[i]*activations[L-2][j];
    }

    for (int l = L-2; l > 0; l--) {
        vector<double> newDelta (sizes[l], 0);
        for (int i = 0; i < weights[l+1].size(); i++) {
            for (int j = 0; j < weights[l].size(); j++) {
                newDelta[j] += delta[i]*weights[l+1][i][j];
                if (i == weights[l+1].size()-1) newDelta[j] *= activationFunctPrime(z[l][j]);
            }
        }
        delta = newDelta;
        updateB[l] = delta;
        for (int i = 0; i < delta.size(); i++) {
            for (int j = 0; j < activations[l-1].size(); j++) updateW[l][i][j] = delta[i]*activations[l-1][j];
        }
    }
    return {updateB, updateW};
}

void Network::save(string filename) {
    ofstream file (filename);

    file << L << "\n";

    // sizes
    for (int i = 0; i < L; i++) file << sizes[i] << " ";
    file << "\n";

    // biases
    for (int i = 1; i < L; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            file << biases[i][j] << " ";
        }
        file << "\n";
    }

    // weights
    for (int i = 1; i < L; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            for (int k = 0; k < sizes[i-1]; k++) {
                file << weights[i][j][k] << " ";
            }
            file << "^";
        }
        file << "\n";
    }

    file.close();
}

void Network::load(string filename) {
    ifstream file (filename);

    file >> L;

    // sizes
    sizes = vector<int> (L);
    for (int i = 0; i < L; i++) file >> sizes[i];

    // biases
    biases = vector<vector<double>> (L);
    for (int i = 1; i < L; i++) {
        biases[i] = vector<double> (sizes[i]);
        for (int j = 0; j < sizes[i]; j++) {
            file >> biases[i][j];
        }
    }

    // weights
    weights = vector<vector<vector<double>>> (L);
    for (int i = 1; i < L; i++) {
        weights[i] = vector<vector<double>> (sizes[i]);
        for (int j = 0; j < sizes[i]; j++) {
            weights[i][j] = vector<double> (sizes[i-1]);
            for (int k = 0; k < sizes[i-1]; k++) {
                file >> weights[i][j][k];
            }
            char c; file >> c;
        }
    }

    file.close();
}