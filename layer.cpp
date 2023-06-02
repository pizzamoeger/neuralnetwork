#include <cassert>
#include "includes.h"

default_random_engine generator;

void fully_connected_layer::init(layer_data data) {

    this->data = data;

    normal_distribution<float> distribution(0.0, 1.0 / sqrt(data.n_in.x));

    biases.resize(data.n_out.x);
    for (int neuron = 0; neuron < data.n_out.x; neuron++) biases[neuron] = distribution(generator);
    biasesVelocity.assign(data.n_out.x, 0);

    weights.resize(data.n_out.x);
    weightsVelocity.resize(data.n_out.x);
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        weights[neuron].resize(data.n_in.x);
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++)
            weights[neuron][previous_neuron] = distribution(generator);
        weightsVelocity[neuron].assign(data.n_in.x, 0);
    }

    updateB = vector<float>(data.n_out.x, 0);
    updateW = vector<vector<float>>(data.n_out.x, vector<float>(data.n_in.x, 0));
}

void fully_connected_layer::feedforward(vector<vector<vector<float>>> &a,
                                        vector<vector<vector<float>>> &derivative_z) {
    vector<vector<vector<float>>> z(1, vector<vector<float>>(1, vector<float>(data.n_out.x, 0)));
    vector<vector<vector<float>>> new_a = z;
    derivative_z = z;
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        // get new activations
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++)
            z[0][0][neuron] += weights[neuron][previous_neuron] * a[0][0][previous_neuron];
        z[0][0][neuron] += biases[neuron];
        new_a[0][0][neuron] = data.activationFunct(z[0][0][neuron]);
        derivative_z[0][0][neuron] = data.activationFunctPrime(z[0][0][neuron]);
    }
    a = new_a;
}

void
fully_connected_layer::backprop(vector<vector<vector<float>>> &delta, vector<vector<vector<float>>> &activations,
                                vector<vector<vector<float>>> &derivative_z) {

    if (!data.last_layer) {
        for (int neuron = 0; neuron < data.n_out.x; neuron++) delta[0][0][neuron] *= derivative_z[0][0][neuron];
    }

    for (int neuron = 0; neuron < data.n_out.x; neuron++) updateB[neuron] += delta[0][0][neuron];

    vector<vector<vector<float>>> newDelta(1, vector<vector<float>>(1, vector<float>(data.n_in.x, 0)));

    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++) {
            updateW[neuron][previous_neuron] += delta[0][0][neuron] * activations[0][0][previous_neuron];
            newDelta[0][0][previous_neuron] += delta[0][0][neuron] * weights[neuron][previous_neuron];
        }
    }
    delta = newDelta;
}

void fully_connected_layer::update(hyperparams params) {
    // update velocities
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        biasesVelocity[neuron] = params.momentum_coefficient * biasesVelocity[neuron] -
                                 (params.fully_connected_biases_learning_rate / params.mini_batch_size) *
                                 updateB[neuron];
    }

    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++) {
            weightsVelocity[neuron][previous_neuron] =
                    params.momentum_coefficient * weightsVelocity[neuron][previous_neuron] -
                    (params.fully_connected_weights_learning_rate / params.mini_batch_size) *
                    updateW[neuron][previous_neuron];
        }
    }

    // update weights and biases
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        biases[neuron] += biasesVelocity[neuron];
    }
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++) {
            weights[neuron][previous_neuron] = (1 - params.fully_connected_weights_learning_rate *
                                                    params.L2_regularization_term / params.training_data_size) *
                                               weights[neuron][previous_neuron] +
                                               weightsVelocity[neuron][previous_neuron];
        }
    }

    updateB = vector<float>(data.n_out.x, 0);
    updateW = vector<vector<float>>(data.n_out.x, vector<float>(data.n_in.x, 0));
}

void convolutional_layer::init(layer_data data) {

    this->data = data;

    this->data.n_out.x = (data.n_in.x - data.receptive_field_length + 1) / data.stride_length;
    this->data.n_out.y = (data.n_in.y - data.receptive_field_length + 1) / data.stride_length;

    normal_distribution<float> distribution(0.0, 1.0 / sqrt(data.receptive_field_length * data.receptive_field_length));

    biases.assign(data.n_out.feature_maps, 0);
    for (int map = 0; map < data.n_out.feature_maps; map++) biases[map] = distribution(generator);
    biasesVelocity.assign(data.n_out.feature_maps, 0);

    weights.resize(data.n_in.feature_maps);
    weightsVelocity.resize(data.n_in.feature_maps);
    for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
        weightsVelocity[previous_map].assign(data.n_out.feature_maps, vector<vector<float>>(data.receptive_field_length,
                                                                                            vector<float>(
                                                                                                    data.receptive_field_length,
                                                                                                    0)));
        weights[previous_map].assign(data.n_out.feature_maps, vector<vector<float>>(data.receptive_field_length,
                                                                                    vector<float>(
                                                                                            data.receptive_field_length,
                                                                                            0)));
        for (int map = 0; map < data.n_out.feature_maps; map++) {
            for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                    weights[previous_map][map][kernel_y][kernel_y] = distribution(generator);
                }
            }
        }
    }

    updateB = vector<float>(data.n_out.feature_maps, 0);
    updateW = vector<vector<vector<vector<float>>>>(data.n_in.feature_maps,
                                                    vector<vector<vector<float>>>(data.n_out.feature_maps,
                                                                                  vector<vector<float>>(
                                                                                          data.receptive_field_length,
                                                                                          vector<float>(
                                                                                                  data.receptive_field_length,
                                                                                                  0))));
}

void convolutional_layer::feedforward(vector<vector<vector<float>>> &a,
                                      vector<vector<vector<float>>> &derivative_z) {

    vector<vector<vector<float>>> z(data.n_out.feature_maps, vector<vector<float>>(data.n_out.y, vector<float>(
            data.n_out.x, 0)));
    vector<vector<vector<float>>> new_a = z;
    derivative_z = z;

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
                    for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                        for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                            z[map][y][x] +=
                                    weights[previous_map][map][kernel_y][kernel_x] *
                                    a[previous_map][y * data.stride_length + kernel_y][
                                            x * data.stride_length + kernel_x];
                        }
                    }
                }
                z[map][y][x] += biases[map];
                new_a[map][y][x] = data.activationFunct(z[map][y][x]);
                derivative_z[map][y][x] = data.activationFunctPrime(z[map][y][x]);
            }
        }
    }

    a = new_a;
}

void convolutional_layer::backprop(vector<vector<vector<float>>> &delta,
                                   vector<vector<vector<float>>> &activations,
                                   vector<vector<vector<float>>> &derivative_z) {

     for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) delta[map][y][x] *= derivative_z[map][y][x];
        }
    }

    vector<vector<vector<float>>> newDelta(data.n_in.feature_maps,
                                           vector<vector<float>>(data.n_in.y, vector<float>(data.n_in.y, 0)));

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                updateB[map] += delta[map][y][x];
                for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
                    for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                        for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                            newDelta[previous_map][y * data.stride_length + kernel_y][x * data.stride_length +
                                                                                      kernel_x] +=
                                    delta[map][y][x] * weights[previous_map][map][kernel_y][kernel_x];
                            updateW[previous_map][map][kernel_y][kernel_x] +=
                                    activations[previous_map][y * data.stride_length + kernel_y][
                                            x * data.stride_length + kernel_x] * delta[map][y][x];
                        }
                    }
                }
            }
        }
    }

    delta = newDelta;
}

void convolutional_layer::update(hyperparams params) {

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        assert(!isnan(biasesVelocity[map]));
        biasesVelocity[map] = params.momentum_coefficient * biasesVelocity[map] -
                              (params.convolutional_biases_learning_rate / params.mini_batch_size) * updateB[map];
        biases[map] += biasesVelocity[map];
    }

    for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
        for (int map = 0; map < data.n_out.feature_maps; map++) {
            for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                    assert(!isnan(weightsVelocity[previous_map][map][kernel_y][kernel_x]));
                    weightsVelocity[previous_map][map][kernel_y][kernel_x] =
                            params.momentum_coefficient * weightsVelocity[previous_map][map][kernel_y][kernel_x] -
                            (params.convolutional_weights_learning_rate / params.mini_batch_size /
                             (data.n_out.x * data.n_out.y) *
                             data.stride_length * data.stride_length) * updateW[previous_map][map][kernel_y][kernel_x];
                    weights[previous_map][map][kernel_y][kernel_x] = (1 -
                                                                      params.convolutional_weights_learning_rate /
                                                                      (data.n_out.x * data.n_out.y) *
                                                                      data.stride_length * data.stride_length *
                                                                      params.L2_regularization_term /
                                                                      params.training_data_size) *
                                                                     weights[previous_map][map][kernel_y][kernel_x] +
                                                                     weightsVelocity[previous_map][map][kernel_y][kernel_x];
                }
            }
        }
    }

    //if ((float)rand()/RAND_MAX > 0.99) cerr << "maximum bias: " << maxb << ", maximum weight: " << maxw << "\n";

    updateB = vector<float>(data.n_out.feature_maps, 0);
    updateW = vector<vector<vector<vector<float>>>>(data.n_in.feature_maps,
                                                    vector<vector<vector<float>>>(data.n_out.feature_maps,
                                                                                  vector<vector<float>>(
                                                                                          data.receptive_field_length,
                                                                                          vector<float>(
                                                                                                  data.receptive_field_length,
                                                                                                  0))));
}

void max_pooling_layer::init(layer_data data) {
    this->data = data;
    this->data.n_out.x = data.n_in.x / data.summarized_region_length;
    this->data.n_out.y = data.n_in.y / data.summarized_region_length;
}

void max_pooling_layer::feedforward(vector<vector<vector<float>>> &a,
                                    vector<vector<vector<float>>> &derivative_z) {
    vector<vector<vector<float>>> z(data.n_out.feature_maps,
                                    vector<vector<float>>(data.n_out.y, vector<float>(data.n_out.x,
                                                                                      numeric_limits<float>::lowest())));
    derivative_z = z;

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                for (int kernel_y = 0; kernel_y < data.summarized_region_length; kernel_y++) {
                    for (int kernel_x = 0; kernel_x < data.summarized_region_length; kernel_x++) {
                        z[map][y][x] = max(z[map][y][x], a[map][y*data.summarized_region_length + kernel_y][x*data.summarized_region_length + kernel_x]);
                    }
                }
                derivative_z[map][y][x] = z[map][y][x];
            }
        }
    }

    a = z;
}

void max_pooling_layer::backprop(vector<vector<vector<float>>> &delta,
                                 vector<vector<vector<float>>> &activations,
                                 vector<vector<vector<float>>> &derivative_z) {
    vector<vector<vector<float>>> newDelta(data.n_in.feature_maps,
                                           vector<vector<float>>(data.n_in.y, vector<float>(data.n_in.y, 0)));

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                for (int kernel_y = 0; kernel_y < data.summarized_region_length; kernel_y++) {
                    for (int kernel_x = 0; kernel_x < data.summarized_region_length; kernel_x++) {
                        if (abs(activations[map][y*data.summarized_region_length  + kernel_y][x*data.summarized_region_length  + kernel_x]-derivative_z[map][y][x]) < pow(10, -8)) {
                            newDelta[map][y*data.summarized_region_length  + kernel_y][x*data.summarized_region_length + kernel_x] = delta[map][y][x];
                        }
                    }
                }
            }
        }
    }

    delta = newDelta;
}

void max_pooling_layer::update(hyperparams params) {
    (void) params;
}

void flatten_layer::init(layer_data data) {
    this->data = data;
    this->data.n_out.x = data.n_in.x * data.n_in.y * data.n_in.feature_maps;
    this->data.n_out.y = 1;
}

void flatten_layer::feedforward(vector<vector<vector<float>>> &a,
                                vector<vector<vector<float>>> &derivative_z) {

    vector<vector<vector<float>>> new_a(1, vector<vector<float>>(1, vector<float>(data.n_out.x, 0)));
    derivative_z = new_a;

    for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
        for (int y = 0; y < data.n_in.y; y++) {
            for (int x = 0; x < data.n_in.x; x++) {
                new_a[0][0][previous_map * data.n_in.y * data.n_in.x + y * data.n_in.x + x] = a[previous_map][y][x];
                derivative_z[0][0][previous_map * data.n_in.y * data.n_in.x + y * data.n_in.x + x] = 1;
            }
        }
    }

    a = new_a;
}

void flatten_layer::backprop(vector<vector<vector<float>>> &delta,
                             vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &derivative_z) {
    (void)derivative_z;
    (void)activations;
    vector<vector<vector<float>>> newDelta(data.n_in.feature_maps,
                                           vector<vector<float>>(data.n_in.y, vector<float>(data.n_in.x, 0)));

    for (int map = 0; map < data.n_in.feature_maps; map++) {
        for (int y = 0; y < data.n_in.y; y++) {
            for (int x = 0; x < data.n_in.x; x++) {
                newDelta[map][y][x] = delta[0][0][map * data.n_in.y * data.n_in.x + y * data.n_in.x + x];
            }
        }
    }

    delta = newDelta;
}

void flatten_layer::update(hyperparams params) {
    (void)params;
}

void input_layer::init(layer_data data) {
    this->data = data;
}

void input_layer::feedforward(vector<vector<vector<float>>> &a,
                              vector<vector<vector<float>>> &z) {
    (void)a;
    (void)z;
}

void input_layer::backprop(vector<vector<vector<float>>> &delta,
                           vector<vector<vector<float>>> &activations,
                           vector<vector<vector<float>>> &z) {
    (void)delta;
    (void)activations;
    (void)z;
}

void input_layer::update(hyperparams params) {
    (void)params;
}