#include "includes.h"

using namespace std;

#define double float

// sigmoid function and its derivative
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}
double sigmoidPrime(double x) {
    return (sigmoid(x)*(1-sigmoid(x)));
}

struct Network {
    // number of layers
    int L;
    // sizes of the layers
    vector<int> sizes;
    // biases[i][j] is bias of jth neuron in ith layer.
    // bias[0] = {}.
    vector<vector<double>> biases;
    vector<vector<double>> biasesVelocity;
    // weights[i][j][k] is weight of jth neuron in ith layer to kth neuron in i-1th layer.
    // weights[0] = {}
    vector<vector<vector<double>>> weights;
    vector<vector<vector<double>>> weightsVelocity;

    Network (vector<int> & sizes, vector<vector<double>> & biases, vector<vector<vector<double>>> & weights) {
        this->L = sizes.size();
        this->sizes = sizes;
        this->biases = biases;
        this->weights = weights;

        biasesVelocity = vector<vector<double>> (L);
        for (int i = 1; i < L; i++) biasesVelocity[i] = vector<double> (sizes[i], 0);
        weightsVelocity = vector<vector<vector<double>>> (L);
        for (int i = 1; i < L; i++) {
            weightsVelocity[i] = vector<vector<double>> (sizes[i]);
            for (int j = 0; j < sizes[i]; j++) {
                weightsVelocity[i][j] = vector<double> (sizes[i-1], 0);
            }
        }
    }

    vector<double> feedforward(vector<double> & a) {
        for (int i = 1; i < L; i++) {
            vector<double> newA (sizes[i], 0);
            for (int j = 0; j < sizes[i]; j++) {
                // get new activations
                for (int k = 0; k < sizes[i-1]; k++) newA[j] += weights[i][j][k]*a[k];
                newA[j] += biases[i][j];
                newA[j] = sigmoid(newA[j]);
            }
            // update a
            a = newA;
        }
        return a;
    }

    void SGD(vector<pair<vector<double>,vector<double>>> training_data, int epochs, int mini_batch_size, double learning_rate, vector<pair<vector<double>, vector<double>>> test_data, double lambda, double momentum_coefficient) {
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

    void update_mini_batch(vector<pair<vector<double>,vector<double>>> & mini_batch, double learning_rate, double lambda, int n, double momentum_coefficient) {
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

    pair<vector<vector<double>>, vector<vector<vector<double>>>> backprop(vector<double> & in, vector<double> & out) {
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
                activations[i][j] = sigmoid(z[i][j]);
            }
        }

        // backpropagate
        auto delta = crossEntropyDelta(activations[L-1], out);
        //for (int i = 0; i < delta.size(); i++) delta[i] *= sigmoidPrime(z[L-1][i]);
        updateB[L-1] = delta;
        for (int i = 0; i < delta.size(); i++) {
            for (int j = 0; j < activations[L-2].size(); j++) updateW[L-1][i][j] = delta[i]*activations[L-2][j];
        }

        for (int l = L-2; l > 0; l--) {
            vector<double> newDelta (sizes[l], 0);
            for (int i = 0; i < weights[l+1].size(); i++) {
                for (int j = 0; j < weights[l].size(); j++) {
                    newDelta[j] += delta[i]*weights[l+1][i][j];
                    if (i==weights[l+1].size()-1) newDelta[j] *= sigmoidPrime(z[l][j]);
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

    // cross entropy cost function
    vector<double> crossEntropyDelta(vector<double> & output_activations, vector<double> & out) {
        vector<double> ret = output_activations;
        for (int i = 0; i < ret.size(); i++) ret[i] -= out[i];
        return ret;
    }
};