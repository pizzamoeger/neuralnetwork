#include <cassert>
#include "includes.h"

default_random_engine generator;

void fully_connected_layer::init (layer_data data, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime) {
    this->n_in = data.n_in.x*data.n_in.y;
    this->n_out = data.n_out.x*data.n_in.y;
    this->activationFunct = activationFunct;
    this->activationFunctPrime = activationFunctPrime;
    this->costFunctPrime = costFunctPrime;

    normal_distribution<double> distribution(0.0, 1.0 / sqrt(n_in));

    biases.resize(n_out);
    for (int neuron = 0; neuron < n_out; neuron++) biases[neuron] = distribution(generator);
    biasesVelocity.assign(n_out, 0);

    weights.resize(n_out);
    weightsVelocity.resize(n_out);
    for (int neuron = 0; neuron < n_out; neuron++) {
        weights[neuron].resize(n_in);
        for (int previous_neuron = 0; previous_neuron < n_in; previous_neuron++) weights[neuron][previous_neuron] = distribution(generator);
        weightsVelocity[neuron].assign(n_in, 0);
    }

    updateB = vector<double> (n_out, 0);
    updateW = vector<vector<double>> (n_out, vector<double> (n_in, 0));
}

void fully_connected_layer::feedforward(int _, vector<vector<vector<double>>> & a, vector<vector<vector<double>>> & z) {
    vector<vector<vector<double>>> new_z (1, vector<vector<double>> (1, vector<double> (n_out, 0)));
    vector<vector<vector<double>>> new_a = new_z;
    for (int neuron = 0; neuron < n_out; neuron++) {
        // get new activations
        for (int previous_neuron = 0; previous_neuron < n_in; previous_neuron++) new_z[0][0][neuron] += weights[neuron][previous_neuron]*a[0][0][previous_neuron];
        new_z[0][0][neuron] += biases[neuron];
        new_a[0][0][neuron] = activationFunct(new_z[0][0][neuron]);
    }
    z = new_z;
    a = new_a;
}

void fully_connected_layer::backprop(int _, vector<double> & delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z) {
    for (int neuron = 0; neuron < n_out; neuron++) updateB[neuron] += delta[neuron];

    for (int neuron = 0; neuron < n_out; neuron++) {
        for (int previous_neuron = 0; previous_neuron < n_in; previous_neuron++) {
            updateW[neuron][previous_neuron] += delta[neuron]*activations[0][0][previous_neuron];
        }
    }

    vector<double> newDelta (n_in, 0);

    for (int neuron = 0; neuron < n_out; neuron++) {
        for (int previous_neuron = 0; previous_neuron < n_in; previous_neuron++) {
            newDelta[previous_neuron] += delta[neuron]*weights[neuron][previous_neuron];
            if (neuron == n_out-1) newDelta[previous_neuron] *= activationFunctPrime(z[0][0][previous_neuron]);
        }
    }
    delta = newDelta;
}

void fully_connected_layer::update(hyperparams params) {
    // update velocities
    for (int neuron = 0; neuron < n_out; neuron++) {
        assert(!isnan(biasesVelocity[neuron]));
        biasesVelocity[neuron] = params.momentum_coefficient * biasesVelocity[neuron] - (params.learning_rate / params.mini_batch_size) * updateB[neuron];
    }

    for (int neuron = 0; neuron < n_out; neuron++) {
        for (int previous_neuron = 0; previous_neuron < n_in; previous_neuron++) {
            assert(!isnan(weightsVelocity[neuron][previous_neuron]));
            weightsVelocity[neuron][previous_neuron] = params.momentum_coefficient * weightsVelocity[neuron][previous_neuron] - (params.learning_rate / params.mini_batch_size) * updateW[neuron][previous_neuron];
        }
    }

    // update weights and biases
    for (int neuron = 0; neuron < n_out; neuron++) {
        biases[neuron] = biases[neuron]+biasesVelocity[neuron];
    }
    for (int neuron = 0; neuron < n_out; neuron++) {
        for (int previous_neuron = 0; previous_neuron < n_in; previous_neuron++) {
            weights[neuron][previous_neuron] = (1-params.learning_rate*params.L2_regularization_term/params.training_data_size)*weights[neuron][previous_neuron]+weightsVelocity[neuron][previous_neuron];
        }
    }

    updateB = vector<double> (n_out, 0);
    updateW = vector<vector<double>> (n_out, vector<double> (n_in, 0));
}

void convolutional_layer::init(layer_data data, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime) {

    this->n_in = data.n_in;

    this->n_out.x = (data.n_in.x-data.receptive_field_length+1)/data.stride_length;
    this->n_out.y = (data.n_in.y-data.receptive_field_length+1)/data.stride_length;

    normal_distribution<double> distribution(0.0, 1.0 / sqrt(n_in.x*n_in.y));

    this->activationFunct = activationFunct;
    this->activationFunctPrime = activationFunctPrime;
    this->costFunctPrime = costFunctPrime;
    this->stride_length = data.stride_length;
    this->receptive_field_length = data.receptive_field_length;
    this->feature_maps = data.feature_maps;

    biases.assign(feature_maps, distribution(generator));
    biasesVelocity.assign(feature_maps, 0);

    weights.resize(feature_maps);
    weightsVelocity.resize(feature_maps);
    for (int map = 0; map < feature_maps; map++) {
        weights[map].assign(receptive_field_length, vector<double> (receptive_field_length, distribution(generator)));
        weightsVelocity[map].assign(receptive_field_length, vector<double> (receptive_field_length, 0));

    }

    updateB = vector<double> (feature_maps, 0);
    updateW = vector<vector<vector<double>>> (feature_maps, vector<vector<double>> (receptive_field_length, vector<double> (receptive_field_length, 0)));
}

void convolutional_layer::feedforward(int previous_feature_maps, vector<vector<vector<double>>> &a, vector<vector<vector<double>>> &z) {

    vector<vector<vector<double>>> new_z (feature_maps*previous_feature_maps, vector<vector<double>> (n_out.y, vector<double> (n_out.x, 0)));
    vector<vector<vector<double>>> new_a = new_z;

    for (int previous_map = 0; previous_map < previous_feature_maps; previous_map++) {
        for (int map = 0; map < feature_maps; map++) {
            for (int y = 0; y < n_out.y; y++) {
                for (int x = 0; x < n_out.x; x++) {
                    for (int kernel_y = 0; kernel_y < receptive_field_length; kernel_y++) {
                        for (int kernel_x = 0; kernel_x < receptive_field_length; kernel_x++) {
                            new_z[previous_map*feature_maps+map][y][x] += weights[map][kernel_y][kernel_x];
                            if (y == n_out.y-1 && x == n_out.x-1) new_z[previous_map*feature_maps+map][y][x] *= a[previous_map][y*stride_length+kernel_y][x*stride_length+kernel_x];
                        }
                    }
                    new_z[previous_map*feature_maps+map][y][x] += biases[map];
                    new_a[previous_map*feature_maps+map][y][x] = activationFunct(new_z[previous_map*feature_maps+map][y][x]);
                }
            }
        }
    }

    z = new_z;
    a = new_a;
}

void convolutional_layer::backprop(int previous_feature_maps, vector<float> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z) {

    for (int map = 0; map < feature_maps; map++) updateB[map] += delta[map];

    for (int map = 0; map < feature_maps; map++) {
        for (int y = 0; y < n_out.y; y++) {
            for (int x = 0; x < n_out.x; x++) {
                for (int kernel_y = 0; kernel_y < receptive_field_length; kernel_y++) {
                    for (int kernel_x = 0; kernel_x < receptive_field_length; kernel_x++) {
                        updateW[map][kernel_y][kernel_x] += weights[map][kernel_y][kernel_x]*activations[map/feature_maps][y*stride_length+kernel_y][x*stride_length+kernel_y];
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
        assert(!isnan(biasesVelocity[map]));
        biasesVelocity[map] = params.momentum_coefficient * biasesVelocity[map] - (params.learning_rate / params.mini_batch_size) * updateB[map];
    }

    for (int map = 0; map < feature_maps; map++) {
        for (int kernel_y = 0; kernel_y < receptive_field_length; kernel_y++) {
            for (int kernel_x = 0; kernel_x < receptive_field_length; kernel_x++) {
                assert(!isnan(weightsVelocity[map][kernel_y][kernel_x]));
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

void max_pooling_layer::init(layer_data data, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime) {
    this->n_in = data.n_in;

    this->n_out.x = data.n_in.x/data.summarized_region_length;
    this->n_out.y = data.n_in.y/data.summarized_region_length;

    this->feature_maps = 1;

    this->summarized_region_length = data.summarized_region_length;
}

void max_pooling_layer::feedforward(int previous_feature_maps, vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &z) {
    vector<vector<vector<double>>> new_z (previous_feature_maps, vector<vector<double>> (n_out.y, vector<double> (n_out.x, numeric_limits<float>::lowest())));
    vector<vector<vector<double>>> new_a = new_z;

    for (int previous_map = 0; previous_map < previous_feature_maps; previous_map++) {
        for (int y = 0; y < n_out.y; y++) {
            for (int x = 0; x < n_out.x; x++) {
                for (int kernel_y = 0; kernel_y < summarized_region_length; kernel_y++) {
                    for (int kernel_x = 0; kernel_x < summarized_region_length; kernel_x++) {
                        new_z[previous_map][y][x] = max(new_z[previous_map][y][x], a[previous_map][y+kernel_y][x+kernel_x]);
                    }
                }
            }
        }
    }

    z = new_z;
    a = new_z;
}

void max_pooling_layer::backprop(int previous_feature_maps, vector<float> &delta,
                                 vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z) {
    vector<float> newDelta (previous_feature_maps, 0);

    for (int previous_map = 0; previous_map < previous_feature_maps; previous_map++) {
        newDelta[previous_map] = n_out.y*n_out.x*summarized_region_length*summarized_region_length*delta[previous_map];
    }

    delta = newDelta;
}

void max_pooling_layer::update(hyperparams params) {}

void flatten_layer::init(layer_data data, const function<float(float)> &activationFunct,
                         const function<float(float)> &activationFunctPrime,
                         const function<float(float, float)> &costFunctPrime) {
    this->n_in = data.n_in;
    this->feature_maps = 1;
    this->n_out.x = data.n_in.x*data.n_in.y;
    this->n_out.y = 1;
}

void flatten_layer::feedforward(int previous_feature_maps, vector<vector<vector<float>>> &a,
                                vector<vector<vector<float>>> &z) {
    vector<vector<vector<double>>> new_z (1, vector<vector<double>> (1, vector<double> (n_out.x*a.size(), 0)));

    for (int previous_map = 0; previous_map < previous_feature_maps; previous_map++) {
        for (int y = 0; y < n_in.y; y++) {
            for (int x = 0; x < n_in.x; x++) {
                new_z[0][0][previous_map*n_in.y*n_in.x+y*n_in.x+x] = a[previous_map][y][x];
            }
        }
    }

    z = new_z;
    a = new_z;
}

void flatten_layer::backprop(int previous_feature_maps, vector<float> &delta,
                             vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z) {
    vector<float> newDelta (previous_feature_maps, 0);

    for (int previous_map = 0; previous_map < previous_feature_maps; previous_map++) {
        for (int y = 0; y < n_in.y; y++) {
            for (int x = 0; x < n_in.x; x++) {
                newDelta[previous_map] += delta[previous_map*n_in.y*n_in.x+y*n_in.x+x];
            }
        }
    }

    delta = newDelta;
}

void flatten_layer::update(hyperparams params) {}

void input_layer::init(layer_data data, const function<float(float)> &activationFunct,
                         const function<float(float)> &activationFunctPrime,
                         const function<float(float, float)> &costFunctPrime) {
    this->n_out = data.n_out;
    this->feature_maps = 1;
}

void input_layer::feedforward(int previous_feature_maps, vector<vector<vector<float>>> &a,
                              vector<vector<vector<float>>> &z) {}
void input_layer::backprop(int previous_feature_maps, vector<float> &delta, vector<vector<vector<float>>> &activations,
                            vector<vector<vector<float>>> &z) {}
void input_layer::update(hyperparams params) {}