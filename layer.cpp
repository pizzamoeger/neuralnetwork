#include "includes.h"

int get_convolutional_weights_index(int previous_map, int map, int y, int x, layer_data &data) {
    return
            previous_map * (data.n_out.feature_maps * data.n_out.x * data.n_out.y)
            + map * (data.n_out.x * data.n_out.y)
            + y * (data.n_out.x)
            + x;
}

int get_fully_connected_weights_index(int neuron, int previous_neuron) {
    return neuron*previous_neuron+previous_neuron;
}

int get_data_index(int map, int y, int x, layer_data &data) {
    return
            map * (data.n_out.x * data.n_out.y)
            + y * (data.n_out.x)
            + x;
}

default_random_engine generator;

void fully_connected_layer::init(layer_data data, layer_data data_previous) {

    data.n_in = {data_previous.n_out.feature_maps * data_previous.n_out.y * data_previous.n_out.x, 1, 1};
    this->data = data;
    this->data_previous = data_previous;
    normal_distribution<float> distribution(0.0, 1.0 / sqrt(data.n_in.x));

    biases.resize(data.n_out.x);
    for (int neuron = 0; neuron < data.n_out.x; neuron++) biases[neuron] = distribution(generator);
    biasesVelocity.assign(data.n_out.x, 0);

    weights.resize(data.n_out.x*data.n_in.x);
    weightsVelocity.resize(data.n_out.x*data.n_in.x);
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++) {
            weights[get_fully_connected_weights_index(neuron, previous_neuron)] = distribution(generator);
            weightsVelocity[get_fully_connected_weights_index(neuron, previous_neuron)] = 0;
        }
    }

    updateB = vector<float>(data.n_out.x, 0);
    updateW = vector<float>(data.n_out.x*data.n_in.x, 0);
}

void fully_connected_layer::feedforward(vector<float> &a,
                                        vector<float> &derivative_z) {
    vector<float> z (data.n_out.x, 0);
    vector<float> new_a = z;
    derivative_z = z;
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        // get new activations
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++)
            z[get_data_index(0, 0, neuron, data)] += weights[get_fully_connected_weights_index(neuron, previous_neuron)] * a[get_data_index(0, 0, previous_neuron, data_previous)];
        z[get_data_index(0, 0, neuron, data)] += biases[neuron];
        new_a[get_data_index(0, 0, neuron, data)] = data.activationFunct(z[get_data_index(0, 0, neuron, data)]);
        derivative_z[get_data_index(0, 0, neuron, data)] = data.activationFunctPrime(z[get_data_index(0, 0, neuron, data)]);
    }
    a = new_a;
}

void
fully_connected_layer::backprop(vector<float> &delta, vector<float> &activations,
                                vector<float> &derivative_z) {

    if (!data.last_layer) {
        for (int neuron = 0; neuron < data.n_out.x; neuron++) delta[get_data_index(0, 0, neuron, data)] *= derivative_z[get_data_index(0, 0, neuron, data)];
    }

    for (int neuron = 0; neuron < data.n_out.x; neuron++) updateB[neuron] += delta[get_data_index(0, 0, neuron, data)];

    vector<float> newDelta (data.n_in.x, 0);

    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++) {
            updateW[get_fully_connected_weights_index(neuron, previous_neuron)] += delta[get_data_index(0, 0, neuron, data)] * activations[get_data_index(0, 0, previous_neuron, data)];
            newDelta[get_data_index(0, 0, previous_neuron, data_previous)] += delta[get_data_index(0, 0, neuron, data)] * weights[get_fully_connected_weights_index(neuron, previous_neuron)];
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
            weightsVelocity[get_fully_connected_weights_index(neuron, previous_neuron)] =
                    params.momentum_coefficient * weightsVelocity[get_fully_connected_weights_index(neuron, previous_neuron)] -
                    (params.fully_connected_weights_learning_rate / params.mini_batch_size) *
                    updateW[get_fully_connected_weights_index(neuron, previous_neuron)];
        }
    }

    // update weights and biases
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        biases[neuron] += biasesVelocity[neuron];
    }
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++) {
            weights[get_fully_connected_weights_index(neuron, previous_neuron)] = (1 - params.fully_connected_weights_learning_rate *
                                                    params.L2_regularization_term / params.training_data_size) *
                                               weights[get_fully_connected_weights_index(neuron, previous_neuron)] +
                                               weightsVelocity[get_fully_connected_weights_index(neuron, previous_neuron)];
        }
    }

    updateB = vector<float>(data.n_out.x, 0);
    updateW = vector<float>(data.n_out.x*data.n_in.x, 0);
}

void fully_connected_layer::save(string filename) {
    ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_FULLY_CONNECTED << "//";
    file << data.n_out.x << "//";

    for (auto bias : biases) file << bias << " ";
    file << "//";
    for (auto biasVel : biasesVelocity) file << biasVel << " ";
    file << "//";
    for (auto weight : weights) file << weight << " ";
    file << "//";
    for (auto weightVec : weightsVelocity) file << weightVec << " ";
    file << "\n";

    file.close();
}

void convolutional_layer::init(layer_data data, layer_data data_previous) {

    data.n_in = data_previous.n_out;
    data.n_out.x = (data.n_in.x - data.receptive_field_length + 1) / data.stride_length;
    data.n_out.y = (data.n_in.y - data.receptive_field_length + 1) / data.stride_length;

    this->data = data;
    this->data_previous = data_previous;

    weights_size = data.n_in.feature_maps * data.n_out.feature_maps * data.n_out.x * data.n_out.y;

    normal_distribution<float> distribution(0.0, 1.0 / sqrt(data.receptive_field_length * data.receptive_field_length));

    biases.assign(data.n_out.feature_maps, 0);
    for (int map = 0; map < data.n_out.feature_maps; map++) biases[map] = distribution(generator);
    biasesVelocity.assign(data.n_out.feature_maps, 0);

    weights.resize(data.n_in.feature_maps);
    weightsVelocity.resize(data.n_in.feature_maps);
    for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
        weightsVelocity.assign(weights_size, 0);
        weights.assign(weights_size, 0);
        for (int map = 0; map < data.n_out.feature_maps; map++) {
            for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                    weights[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_y, data)] = distribution(generator);
                }
            }
        }
    }

    updateB = vector<float>(data.n_out.feature_maps, 0);
    updateW = vector<float> (weights_size, 0);
}

void convolutional_layer::feedforward(vector<float> &a,
                                      vector<float> &derivative_z) {

    vector<float> z (data.n_out.feature_maps * data.n_out.y * data.n_out.x, 0);
    vector<float> new_a = z;
    derivative_z = z;

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
                    for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                        for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                            z[get_data_index(map, y, x, data)] +=
                                    weights[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] *
                                    a[get_data_index(previous_map, y * data.stride_length + kernel_y, x * data.stride_length + kernel_x, data)];
                        }
                    }
                }
                z[get_data_index(map, y, x, data)] += biases[map];
                new_a[get_data_index(map, y, x, data)] = data.activationFunct(z[get_data_index(map, y, x, data)]);
                derivative_z[get_data_index(map, y, x, data)] = data.activationFunctPrime(z[get_data_index(map, y, x, data)]);
            }
        }
    }

    a = new_a;
}

void convolutional_layer::backprop(vector<float> &delta,
                                   vector<float> &activations,
                                   vector<float> &derivative_z) {

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) delta[get_data_index(map, y, x, data)] *= derivative_z[get_data_index(map, y, x, data)];
        }
    }

    vector<float> newDelta(data.n_in.feature_maps * data.n_in.y *data.n_in.y, 0);

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                updateB[map] += delta[get_data_index(map, y, x, data)];
                for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
                    for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                        for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                            newDelta[get_data_index(previous_map, y * data.stride_length + kernel_y, x * data.stride_length +
                                                                                      kernel_x, data_previous)] +=
                                    delta[get_data_index(map, y, x, data)] * weights[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)];
                            updateW[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] +=
                                    activations[get_data_index(previous_map, y * data.stride_length + kernel_y,
                                            x * data.stride_length + kernel_x, data)] * delta[get_data_index(map, y, x, data)];
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
        biasesVelocity[map] = params.momentum_coefficient * biasesVelocity[map] -
                              (params.convolutional_biases_learning_rate / params.mini_batch_size) * updateB[map];
        biases[map] += biasesVelocity[map];
    }

    for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
        for (int map = 0; map < data.n_out.feature_maps; map++) {
            for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                    weightsVelocity[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] =
                            params.momentum_coefficient * weightsVelocity[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] -
                            (params.convolutional_weights_learning_rate / params.mini_batch_size /
                             (data.n_out.x * data.n_out.y) *
                             data.stride_length * data.stride_length) * updateW[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)];
                    weights[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] = (1 -
                                                                      params.convolutional_weights_learning_rate /
                                                                      (data.n_out.x * data.n_out.y) *
                                                                      data.stride_length * data.stride_length *
                                                                      params.L2_regularization_term /
                                                                      params.training_data_size) *
                                                                     weights[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] +
                                                                     weightsVelocity[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)];
                }
            }
        }
    }

    updateB = vector<float>(data.n_out.feature_maps, 0);
    updateW = vector<float>(weights_size, 0);
}

void convolutional_layer::save(string filename) {
    ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_CONVOLUTIONAL << "//";
    file << data.stride_length << " " << data.receptive_field_length << " " << data.n_out.feature_maps << "//";

    for (auto bias : biases) file << bias << " ";
    file << "//";
    for (auto biasVel : biasesVelocity) file << biasVel << " ";
    file << "//";
    for (auto weight : weights) file << weight << " ";
    file << "//";
    for (auto weightVel : weightsVelocity) file << weightVel << " ";
    file << "\n";

    file.close();
}

void max_pooling_layer::init(layer_data data, layer_data data_previous) {
    data.n_in = data_previous.n_out;
    this->data = data;
    this->data_previous = data_previous;
    this->data.n_out.x = data.n_in.x / data.summarized_region_length;
    this->data.n_out.y = data.n_in.y / data.summarized_region_length;
    this->data.n_out.feature_maps = data_previous.n_out.feature_maps;
}

void max_pooling_layer::feedforward(vector<float> &a,
                                    vector<float> &derivative_z) {
    vector<float> z(data.n_out.feature_maps * data.n_out.y * data.n_out.x,numeric_limits<float>::lowest());
    derivative_z = z;

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                for (int kernel_y = 0; kernel_y < data.summarized_region_length; kernel_y++) {
                    for (int kernel_x = 0; kernel_x < data.summarized_region_length; kernel_x++) {
                        z[get_data_index(map, y, x, data)] = max(z[get_data_index(map, y, x, data)], a[get_data_index(map, y * data.summarized_region_length + kernel_y, x * data.summarized_region_length + kernel_x, data)]);
                    }
                }
                derivative_z[get_data_index(map, y, x, data)] = z[get_data_index(map, y, x, data)];
            }
        }
    }

    a = z;
}

void max_pooling_layer::backprop(vector<float> &delta,
                                 vector<float> &activations,
                                 vector<float> &derivative_z) {
    vector<float> newDelta(data.n_in.feature_maps * data.n_in.y * data.n_in.y, 0);

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                for (int kernel_y = 0; kernel_y < data.summarized_region_length; kernel_y++) {
                    for (int kernel_x = 0; kernel_x < data.summarized_region_length; kernel_x++) {
                        if (abs(activations[get_data_index(map, y * data.summarized_region_length + kernel_y,
                                        x * data.summarized_region_length + kernel_x, data_previous)] - derivative_z[get_data_index(map, y, x, data)]) <
                            pow(10, -8)) {
                            newDelta[get_data_index(map, y * data.summarized_region_length + kernel_y,
                                    x * data.summarized_region_length + kernel_x, data_previous)] = delta[get_data_index(map, y, x, data)];
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

void max_pooling_layer::save(string filename) {
    ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_MAX_POOLING << "//";
    file << data.summarized_region_length << "\n";

    file.close();
}

void input_layer::init(layer_data data, layer_data data_previous) {
    this->data = data;
    (void) data_previous;
}

void input_layer::feedforward(vector<float> &a,
                              vector<float> &z) {
    (void) a;
    (void) z;
}

void input_layer::backprop(vector<float> &delta,
                           vector<float> &activations,
                           vector<float> &z) {
    (void) delta;
    (void) activations;
    (void) z;
}

void input_layer::update(hyperparams params) {
    (void) params;
}

void input_layer::save(string filename) {
    ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_INPUT << "//";
    file << data.n_out.x << " " << data.n_out.y << "\n";

    file.close();
}