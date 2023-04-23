#include "includes.h"

default_random_engine generator;

void fully_connected_layer::init (int n_in, int n_out, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime) {
    normal_distribution<double> distribution(0.0, 1.0 / sqrt(n_in));

    this->n_in = n_in;
    this->n_out = n_out;
    this->activationFunct = activationFunct;
    this->activationFunctPrime = activationFunctPrime;
    this->costFunctPrime = costFunctPrime;

    biases.assign(n_out, distribution(generator));
    biasesVelocity.assign(n_out, 0);

    weights.resize(n_out);
    weightsVelocity.resize(n_out);
    for (int i = 0; i < n_out; i++) {
        weights[i].assign(n_in, distribution(generator));
        weightsVelocity[i].assign(n_in, 0);
    }

    updateB = vector<double> (n_out, 0);
    updateW = vector<vector<double>> (n_out, vector<double> (n_in, 0));
}

vector<double> fully_connected_layer::feedforward(vector<double> & a) {
    vector<double> z (n_out, 0);
    for (int i = 0; i < n_out; i++) {
        // get new activations
        for (int k = 0; k < n_in; k++) z[i] += weights[i][k]*a[k];
        z[i] += biases[i];
    }
    return z;
}

void fully_connected_layer::backprop(vector<double> & delta, vector<double> & activations, vector<double> & z) {

    for (int i = 0; i < n_out; i++) updateB[i] += delta[i];

    for (int i = 0; i < n_out; i++) {
        for (int j = 0; j < n_in; j++) {
            updateW[i][j] += delta[i]*activations[j];
        }
    }

    vector<double> newDelta (n_in, 0);

    for (int i = 0; i < n_out; i++) {
        for (int j = 0; j < n_in; j++) {
            newDelta[j] += delta[i]*weights[i][j]*activationFunctPrime(z[j]);;
        }
    }
    delta = newDelta;
}

void fully_connected_layer::update(hyperparams params) {
    // update velocities
    for (int j = 0; j < n_out; j++) {
        biasesVelocity[j] = params.momentum_coefficient * biasesVelocity[j] - (params.learning_rate / params.mini_batch_size) * updateB[j];
    }

    for (int j = 0; j < n_out; j++) {
        for (int k = 0; k < n_in; k++) {
            weightsVelocity[j][k] = params.momentum_coefficient * weightsVelocity[j][k] - (params.learning_rate / params.mini_batch_size) * updateW[j][k];
        }
    }

    // update weights and biases
    for (int j = 0; j < n_out; j++) {
        biases[j] = biases[j]+biasesVelocity[j];
    }
    for (int j = 0; j < n_out; j++) {
        for (int k = 0; k < n_in; k++) {
            weights[j][k] = (1-params.learning_rate*params.L2_regularization_term/params.training_data_size)*weights[j][k]+weightsVelocity[j][k];
        }
    }

    updateB = vector<double> (n_out, 0);
    updateW = vector<vector<double>> (n_out, vector<double> (n_in, 0));
}

void convolutional_layer::init(network_data n_in, int stride_length, int receptive_field_length, int feature_maps,
                               const function<float(float)> &activationFunct,
                               const function<float(float)> &activationFunctPrime,
                               const function<float(float, float)> &costFunctPrime) {
    normal_distribution<double> distribution(0.0, 1.0 / sqrt(n_in.x*n_in.y));

    this->n_in = n_in;

    this->n_out.x = (n_in.x-receptive_field_length+1)/stride_length;
    this->n_out.y = (n_in.y-receptive_field_length+1)/stride_length;

    this->activationFunct = activationFunct;
    this->activationFunctPrime = activationFunctPrime;
    this->costFunctPrime = costFunctPrime;
    this->stride_length = stride_length;
    this->receptive_field_length = receptive_field_length;
    this->feature_maps = feature_maps;

    biases.assign(feature_maps, distribution(generator));
    biasesVelocity.assign(feature_maps, 0);

    weights.resize(feature_maps);
    weightsVelocity.resize(feature_maps);
    for (int i = 0; i < feature_maps; i++) {
        weights[i].assign(receptive_field_length, vector<double> (receptive_field_length, distribution(generator)));
        weightsVelocity[i].assign(receptive_field_length, vector<double> (receptive_field_length, 0));

    }

    updateB = vector<double> (feature_maps, 0);
    updateW = vector<vector<vector<double>>> (feature_maps, vector<vector<double>> (receptive_field_length, vector<double> (receptive_field_length, 0)));
}

pair<vector<vector<vector<double>>>,vector<vector<vector<double>>>> convolutional_layer::feedforward(int previous_feature_maps, vector<vector<vector<double>>> &a) {

    vector<vector<vector<double>>> z (feature_maps*previous_feature_maps, vector<vector<double>> (n_out.y, vector<double> (n_out.y, 0)));
    vector<vector<vector<double>>> new_a = z;

    for (int previous_map = 0; previous_map < previous_feature_maps; previous_map++) {
        for (int map = 0; map < feature_maps; map++) {
            for (int y = 0; y < n_out.y; y++) {
                for (int x = 0; x < n_out.x; x++) {
                    for (int kernel_y = 0; kernel_y < receptive_field_length; kernel_y++) {
                        for (int kernel_x = 0; kernel_x < receptive_field_length; kernel_y++) {
                            z[previous_map*feature_maps+map][y][x] += weights[map][kernel_y][kernel_x]*a[previous_map][y*stride_length+kernel_y][x*stride_length+kernel_x];
                        }
                    }
                    z[previous_map*feature_maps+map][y][x] += biases[map];
                    new_a[previous_map*feature_maps+map][y][x] = activationFunct(z[previous_map*feature_maps+map][y][x]);
                }
            }
        }
    }

    return {new_a, z};
}

void convolutional_layer::backprop(int previous_feature_maps, vector<float> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z) {

    for (int map = 0; map < feature_maps; map++) updateB[map] += delta[map];

    for (int map = 0; map < feature_maps; map++) {
        for (int y = 0; y < n_out.y; y++) {
            for (int x = 0; x < n_out.x; x++) {
                for (int kernel_y = 0; kernel_y < receptive_field_length; kernel_y++) {
                    for (int kernel_x = 0; kernel_x < receptive_field_length; kernel_x++) {
                        updateW[map][kernel_y][kernel_x] += weights[map][kernel_y][kernel_x]*activations[map/previous_feature_maps][y*stride_length+kernel_y][x*stride_length+kernel_y];
                    }
                }
            }
        }
    }

    vector<float> newDelta (previous_feature_maps, 0);

    for (int previous_map = 0; previous_map < previous_feature_maps; previous_map++) {
        for (int map = 0; map < feature_maps; map++) {
            for (int y = 0; y < n_out.y; y++) {
                for (int x = 0; x < n_out.x; x++) {
                    for (int kernel_y = 0; kernel_y < receptive_field_length; kernel_y++) {
                        for (int kernel_x = 0; kernel_x < receptive_field_length; kernel_x++) {
                            newDelta[previous_map] += delta[previous_map*feature_maps+map]*weights[map][kernel_y][kernel_x]*activationFunctPrime(z[previous_map][y*stride_length+kernel_y][x*stride_length+kernel_x]);
                        }
                    }
                }
            }
        }
    }

    delta = newDelta;
}

void convolutional_layer::update (hyperparams params) {
    // update velocities
    for (int map = 0; map < feature_maps; map++) {
        biasesVelocity[map] = params.momentum_coefficient * biasesVelocity[map] - (params.learning_rate / params.mini_batch_size) * updateB[map];
    }

    for (int map = 0; map < feature_maps; map++) {
        for (int kernel_y = 0; kernel_y < receptive_field_length; kernel_y++) {
            for (int kernel_x = 0; kernel_x < receptive_field_length; kernel_x++) {
                weightsVelocity[map][kernel_y][kernel_x] = params.momentum_coefficient * weightsVelocity[map][kernel_y][kernel_x] - (params.learning_rate / params.mini_batch_size) * updateW[map][kernel_y][kernel_x];
            }
        }
    }

    // update weights and biases
    for (int map = 0; map < feature_maps; map++) {
        biases[map] = biases[map]+biasesVelocity[map];
    }

    for (int map = 0; map < feature_maps; map++) {
        for (int kernel_y = 0; kernel_y < receptive_field_length; kernel_y++) {
            for (int kernel_x = 0; kernel_x < receptive_field_length; kernel_x++) {
                weights[map][kernel_y][kernel_x] = (1-params.learning_rate*params.L2_regularization_term/params.training_data_size)*weights[map][kernel_y][kernel_x]+weightsVelocity[map][kernel_y][kernel_x];
            }
        }
    }

    updateB = vector<double> (feature_maps, 0);
    updateW = vector<vector<vector<double>>> (feature_maps, vector<vector<double>> (receptive_field_length, vector<double> (receptive_field_length, 0)));
}