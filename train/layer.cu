#include "includes.h"

int get_convolutional_weights_index(int previous_map, int map, int y, int x, layer_data &data) {
    return
            previous_map * (data.n_out.feature_maps * data.receptive_field_length * data.receptive_field_length)
            + map * (data.receptive_field_length * data.receptive_field_length)
            + y * (data.receptive_field_length)
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

random_device rd;
default_random_engine generator(rd());

void fully_connected_layer::init(layer_data data, layer_data data_previous) {

    data.n_in = {data_previous.n_out.feature_maps * data_previous.n_out.y * data_previous.n_out.x, 1, 1};
    this->data = data;
    this->data_previous = data_previous;
    normal_distribution<float> distribution(0.0, 1.0 / sqrt(data.n_in.x));

    biases = new float [data.n_out.x];
    biasesVelocity = new float [data.n_out.x];
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        biases[neuron] = distribution(generator);
        biasesVelocity[neuron] = 0;
    }

    weights = new float[data.n_out.x*data.n_in.x];
    weightsVelocity = new float[data.n_out.x*data.n_in.x];
    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++) {
            weights[get_fully_connected_weights_index(neuron, previous_neuron)] = distribution(generator);
            weightsVelocity[get_fully_connected_weights_index(neuron, previous_neuron)] = 0;
        }
    }

    updateB = new float [data.n_out.x];
    updateW = new float [data.n_out.x*data.n_in.x];
    for (int bias = 0; bias < data.n_out.x; bias++) updateB[bias] = 0;
    for (int weight = 0; weight < data.n_out.x*data.n_in.x; weight++) updateW[weight] = 0;
}

void fully_connected_layer::feedforward(float* a, float* dz, float* &new_a, float* &new_dz) {
    (void) dz;
    float* z = new float [data.n_out.x];
    for (int i = 0; i < data.n_out.x; i++) z[i] = 0;

    for (int neuron = 0; neuron < data.n_out.x; neuron++) {
        // get new activations
        for (int previous_neuron = 0; previous_neuron < data.n_in.x; previous_neuron++)
            z[neuron] += weights[get_fully_connected_weights_index(neuron, previous_neuron)] * a[previous_neuron];
        z[neuron] += biases[neuron];
        new_a[neuron] = data.activationFunct(z[neuron]);
        new_dz[neuron] = data.activationFunctPrime(z[neuron]);
    }

    delete[] z;
}

void
fully_connected_layer::backprop(vector<float> &delta, float* &activations, float* &derivative_z) {

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

    for (int bias = 0; bias < data.n_out.x; bias++) updateB[bias] = 0;
    for (int weight = 0; weight < data.n_out.x*data.n_in.x; weight++) updateW[weight] = 0;
}

void fully_connected_layer::save(string filename) {
    ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_FULLY_CONNECTED << "//";
    file << data.n_out.x << "//";

    for (int bias = 0; bias < data.n_out.x; bias++) file << biases[bias] << " ";
    file << "//";
    for (int biasVel = 0; biasVel < data.n_out.x; biasVel++) file << biasesVelocity[biasVel] << " ";
    file << "//";
    for (int weight = 0; weight < data.n_out.x*data.n_in.x; weight++) file << weights[weight] << " ";
    file << "//";
    for (int weightVel = 0; weightVel < data.n_out.x*data.n_in.x; weightVel++) file << weightsVelocity[weightVel] << " ";
    file << "\n";

    file.close();
}

void fully_connected_layer::clear() {
    delete[] weights;
    delete[] weightsVelocity;
    delete[] biases;
    delete[] biasesVelocity;
    delete[] updateW;
    delete[] updateB;
}

void convolutional_layer::init(layer_data data, layer_data data_previous) {

    data.n_in = data_previous.n_out;
    data.n_out.x = (data.n_in.x - data.receptive_field_length + 1) / data.stride_length;
    data.n_out.y = (data.n_in.y - data.receptive_field_length + 1) / data.stride_length;

    this->data = data;
    this->data_previous = data_previous;

    weights_size = data.n_in.feature_maps * data.n_out.feature_maps * data.receptive_field_length * data.receptive_field_length;

    normal_distribution<float> distribution(0.0, 1.0 / sqrt(data.receptive_field_length * data.receptive_field_length));

    biases = new float[data.n_out.feature_maps];
    biasesVelocity = new float[data.n_out.feature_maps];
    for (int map = 0; map < data.n_out.feature_maps; map++) {
        biases[map] = distribution(generator);
        biasesVelocity[map] = 0;
    }

    weights = new float[weights_size];
    weightsVelocity = new float[weights_size];
    for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
        for (int map = 0; map < data.n_out.feature_maps; map++) {
            for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                    weights[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] = distribution(generator);
                    weightsVelocity[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] = 0;
                }
            }
        }
    }

    updateB = new float[data.n_out.feature_maps];
    updateW = new float[weights_size];
    for (int bias = 0; bias < data.n_out.feature_maps; bias++) updateB[bias] = 0;
    for (int weight = 0; weight < weights_size; weight++) updateW[weight] = 0;
}

void convolutional_layer::feedforward(float* a, float* dz, float* &new_a, float* &new_dz) {
    (void) dz;

    float* z = new float [data.n_out.feature_maps * data.n_out.y * data.n_out.x];
    for (int i = 0; i < data.n_out.feature_maps * data.n_out.y * data.n_out.x; i++) z[i] = 0;

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
                    for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                        for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                            /*assert(get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data) < weights_size);
                            assert(get_data_index(previous_map, y * data.stride_length + kernel_y, x * data.stride_length + kernel_x, data_previous) < data_previous.n_out.feature_maps * data_previous.n_out.x * data_previous.n_out.y);
                            if (weights[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] > 0) rn++;*/
                            z[get_data_index(map, y, x, data)] +=
                                    weights[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] *
                                    a[get_data_index(previous_map, y * data.stride_length + kernel_y, x * data.stride_length + kernel_x, data_previous)];
                        }
                    }
                }
                //if (biases[map] > 0.5) rn++;
                z[get_data_index(map, y, x, data)] += biases[map];
                new_a[get_data_index(map, y, x, data)] = data.activationFunct(z[get_data_index(map, y, x, data)]);
                new_dz[get_data_index(map, y, x, data)] = data.activationFunctPrime(z[get_data_index(map, y, x, data)]);
            }
        }
    }

    //cout << new_a[get_data_index(data.n_out.feature_maps-1, data.n_out.y-1, data.n_out.x-1, data)] << " ";

    delete[] z;
}

void convolutional_layer::backprop(vector<float> &delta,
                                   float* &activations,
                                   float* &derivative_z) {

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

    for (int i = 0; i < data.n_out.feature_maps; i++) updateB[i] = 0;
    for (int i = 0; i < data.n_in.feature_maps * data.n_out.feature_maps * data.receptive_field_length * data.receptive_field_length; i++) updateW[i] = 0;
}

void convolutional_layer::save(string filename) {
    ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_CONVOLUTIONAL << "//";
    file << data.stride_length << " " << data.receptive_field_length << " " << data.n_out.feature_maps << "//";

    for (int bias = 0; bias < data.n_out.feature_maps; bias++) file << biases[bias] << " ";
    file << "//";
    for (int biasVel = 0; biasVel < data.n_out.feature_maps; biasVel++) file << biasesVelocity[biasVel] << " ";
    file << "//";
    for (int weight = 0; weight < weights_size; weight++) file << weights[weight] << " ";
    file << "//";
    for (int weightVel = 0; weightVel < weights_size; weightVel++) file << weightsVelocity[weightVel] << " ";
    file << "\n";

    file.close();
}

void convolutional_layer::clear() {
    delete[] weights;
    delete[] weightsVelocity;
    delete[] biases;
    delete[] biasesVelocity;
    delete[] updateW;
    delete[] updateB;
}

void max_pooling_layer::init(layer_data data, layer_data data_previous) {
    data.n_in = data_previous.n_out;
    this->data = data;
    this->data_previous = data_previous;
    this->data.n_out.x = data.n_in.x / data.summarized_region_length;
    this->data.n_out.y = data.n_in.y / data.summarized_region_length;
    this->data.n_out.feature_maps = data_previous.n_out.feature_maps;
}

void max_pooling_layer::feedforward(float* a, float* dz, float* &new_a, float* &new_dz) {
    (void) dz;

    for (int i = 0; i < data.n_out.feature_maps * data.n_out.y * data.n_out.x; i++) new_a[i] = numeric_limits<float>::lowest();

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                for (int kernel_y = 0; kernel_y < data.summarized_region_length; kernel_y++) {
                    for (int kernel_x = 0; kernel_x < data.summarized_region_length; kernel_x++) {
                        new_a[get_data_index(map, y, x, data)] = max(new_a[get_data_index(map, y, x, data)], a[get_data_index(map, y * data.summarized_region_length + kernel_y, x * data.summarized_region_length + kernel_x, data_previous)]);
                    }
                }
                new_dz[get_data_index(map, y, x, data)] = new_a[get_data_index(map, y, x, data)];
            }
        }
    }
}

void max_pooling_layer::backprop(vector<float> &delta,
                                 float* &activations, float* &derivative_z) {
    vector<float> newDelta(data.n_in.feature_maps * data.n_in.y * data.n_in.y, 0);
    const float epsilon = 1e-8;

    //cout << activations[get_data_index(data.n_out.feature_maps-1, (data.n_out.y-1)*data.summarized_region_length+data.summarized_region_length-1, (data.n_out.x-1)*data.summarized_region_length+data.summarized_region_length-1, data_previous)] << "sdfkjdslksfjlsf\n";
    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                for (int kernel_y = 0; kernel_y < data.summarized_region_length; kernel_y++) {
                    for (int kernel_x = 0; kernel_x < data.summarized_region_length; kernel_x++) {
                        int act = activations[get_data_index(map, y * data.summarized_region_length + kernel_y, x * data.summarized_region_length + kernel_x, data_previous)];
                        int dev = derivative_z[get_data_index(map, y, x, data)];
                        if (act < dev) swap(act, dev);
                        if (act - dev < epsilon) {
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

void max_pooling_layer::clear() {}

void input_layer::init(layer_data data, layer_data data_previous) {
    this->data = data;
    (void) data_previous;
}

void input_layer::feedforward(float* a, float* dz, float* &new_a, float* &new_dz) {
    (void) a;
    (void) dz;
    (void) new_a;
    (void) new_dz;
}

void input_layer::backprop(vector<float> &delta,
                           float* &activations, float* &derivative_z) {
    (void) delta;
    (void) activations;
    (void) derivative_z;
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

void input_layer::clear() {}