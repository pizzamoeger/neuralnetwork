#include "includes.h"

random_device rd;
default_random_engine generator(rd());

void fully_connected_layer::init(layer_data data, layer_data data_previous) {

    data.n_in = {data_previous.n_out.feature_maps * data_previous.n_out.y * data_previous.n_out.x, 1, 1};
    this->data = data;
    this->data_previous = data_previous;
    int* data_n_in_x;
    cudaMalloc((void**) &data_n_in_x, sizeof(int));
    cudaMemcpy(data_n_in_x, &data.n_in.x, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &dev_weights, data.n_out.x*data.n_in.x*sizeof(float));
    cudaMalloc((void**) &dev_weights_vel, data.n_out.x*data.n_in.x*sizeof(float));
    cudaMalloc((void**) &dev_weights_updt, data.n_out.x*data.n_in.x*sizeof(float));
    set_to_random<<<data.n_out.x * data.n_in.x, 1>>>(dev_weights, data_n_in_x);
    set_to<<<data.n_out.x * data.n_in.x, 1>>>(dev_weights_vel, 0);
    set_to<<<data.n_out.x * data.n_in.x, 1>>>(dev_weights_updt, 0);

    cudaMalloc((void**) &dev_biases, data.n_out.x*sizeof(float));
    cudaMalloc((void**) &dev_biases_vel, data.n_out.x*sizeof(float));
    cudaMalloc((void**) &dev_biases_updt, data.n_out.x*sizeof(float));
    set_to_random<<<data.n_out.x, 1>>>(dev_biases, data_n_in_x);
    set_to<<<data.n_out.x,1>>>(dev_biases_vel, 0);
    set_to<<<data.n_out.x,1>>>(dev_biases_updt, 0);
}

void fully_connected_layer::feedforward(float* a, float* dz, float* &new_a, float* &new_dz) {
    (void) dz;

    float* dev_a;
    float* dev_z;
    float* dev_new_a;
    float* dev_new_dz;
    int* dev_n_in_x;

    cudaMalloc((void**) &dev_n_in_x, sizeof(int));
    cudaMalloc((void**) &dev_a, data.n_in.x*sizeof(float));
    cudaMalloc((void**) &dev_z, data.n_out.x*sizeof(float));
    cudaMalloc((void**) &dev_new_a, data.n_out.x*sizeof(float));
    cudaMalloc((void**) &dev_new_dz, data.n_out.x*sizeof(float));
    cudaDeviceSynchronize();

    set_to<<<data.n_out.x, 1>>>(dev_z, 0);
    cudaMemcpy(dev_a, a, data.n_in.x*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_n_in_x, &data.n_in.x, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    dim3 grid(data.n_out.x, data.n_in.x);
    addWeights<<<grid, 1>>>(dev_a, dev_weights, dev_z, dev_n_in_x);
    cudaDeviceSynchronize();
    getNewA<<<data.n_out.x,1>>> (dev_z, dev_biases, dev_new_a, dev_new_dz);
    cudaDeviceSynchronize();
    cudaMemcpy(new_a, dev_new_a, data.n_out.x*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_dz, dev_new_dz, data.n_out.x*sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(dev_z);
    cudaFree(dev_a);
    cudaFree(dev_new_a);
    cudaFree(dev_new_dz);
    cudaFree(dev_n_in_x);
}

void fully_connected_layer::backprop(float * &delta, float* &activations, float* &derivative_z, float * &new_delta) {
    float* dev_delta;
    float* dev_derivative_z;
    float* dev_activations;
    float* dev_new_delta;
    int* dev_data_n_in_x;

    cudaError err = cudaMalloc((void**) &dev_new_delta, data.n_in.x*sizeof(float));
    if (err != cudaSuccess) cerr << cudaGetErrorString(err) << "dev new delta\n";
    err = cudaMalloc((void**) &dev_delta, data.n_out.x*sizeof(float));
    //if (err != cudaSuccess) cout << cudaGetErrorString(err) << "dev delta\n";
    err = cudaMalloc((void**) &dev_derivative_z, data.n_out.x*sizeof(float));
    //if (err != cudaSuccess) cout << cudaGetErrorString(err) << "dev dz\n";
    cudaMemcpy(dev_delta, delta, data.n_out.x*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_derivative_z, derivative_z, data.n_out.x*sizeof(float), cudaMemcpyHostToDevice);
    set_to<<<data.n_in.x,1>>>(dev_new_delta, 0);
    cudaMalloc((void**) &dev_activations, data.n_in.x*sizeof(float));
    cudaMemcpy(dev_activations, activations, data.n_in.x*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_data_n_in_x, sizeof(int));
    cudaMemcpy(dev_data_n_in_x, &data.n_in.x, sizeof(int), cudaMemcpyHostToDevice);

    if (!data.last_layer) {
        mult<<<data.n_out.x,1>>>(dev_delta, dev_derivative_z);
    }
    /*cudaDeviceSynchronize();
    for (int i = 0; i < data.n_out.x; i++) cerr << delta[i]*derivative_z[i] << " ";
    cerr << "\n";
    cudaMemcpy(delta, dev_delta, data.n_out.x*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < data.n_out.x; i++) cerr << delta[i] << " ";
    cerr << "\n";*/
    // BUG FREE HERE

    add<<<data.n_out.x,1>>>(dev_biases_updt, dev_delta);
    cudaDeviceSynchronize();

    dim3 grid(data.n_out.x, data.n_in.x);
    backprop_logic<<<grid,1>>>(dev_weights_updt, dev_delta, dev_activations, dev_new_delta, dev_weights, dev_data_n_in_x);
    cudaDeviceSynchronize();

    cudaMemcpy(new_delta, dev_new_delta, data.n_in.x*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_delta);
    cudaFree(dev_derivative_z);
    cudaFree(dev_activations);
    cudaFree(dev_new_delta);
    cudaFree(dev_data_n_in_x);
}

void fully_connected_layer::update(hyperparams params) {
    // update velocities
    hyperparams* dev_params;
    cudaMalloc((void**) &dev_params, sizeof(hyperparams));
    cudaMemcpy(dev_params, &params, sizeof(hyperparams), cudaMemcpyHostToDevice);

    update_bias_vel<<<data.n_out.x,1>>>(dev_biases_vel, dev_biases_updt, dev_params);
    update_weights_vel<<<data.n_out.x*data.n_in.x,1>>>(dev_weights_vel, dev_weights_updt, dev_params);
    cudaDeviceSynchronize();

    // update weights and biases
    add<<<data.n_out.x,1>>>(dev_biases, dev_biases_vel);
    update_weights<<<data.n_out.x*data.n_in.x,1>>>(dev_weights, dev_weights_vel, dev_params);
    cudaDeviceSynchronize();

    set_to<<<data.n_out.x,1>>>(dev_biases_updt, 0);
    set_to<<<data.n_out.x*data.n_in.x,1>>>(dev_weights_updt, 0);
    cudaDeviceSynchronize();

    cudaFree(dev_params);
}

void fully_connected_layer::save(string filename) {
    ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_FULLY_CONNECTED << "//";
    file << data.n_out.x << "//";

    float* biases = new float [data.n_out.x];
    cudaMemcpy(biases, dev_biases, data.n_out.x*sizeof(float), cudaMemcpyDeviceToHost);
    for (int bias = 0; bias < data.n_out.x; bias++) file << biases[bias] << " ";
    delete[] biases;
    file << "//";

    float* biases_vel = new float [data.n_out.x];
    cudaMemcpy(biases_vel, dev_biases, data.n_out.x*sizeof(float), cudaMemcpyDeviceToHost);
    for (int bias_vel = 0; bias_vel < data.n_out.x; bias_vel++) file << biases_vel[bias_vel] << " ";
    delete[] biases_vel;
    file << "//";

    float* weights = new float [data.n_out.x*data.n_in.x];
    cudaMemcpy(weights, dev_weights, data.n_out.x*data.n_in.x*sizeof(float), cudaMemcpyDeviceToHost);
    for (int weight = 0; weight < data.n_out.x*data.n_in.x; weight++) file << weights[weight] << " ";
    delete[] weights;
    file << "//";

    float* weights_vel = new float [data.n_out.x*data.n_in.x];
    cudaMemcpy(weights_vel, dev_weights_vel, data.n_out.x*data.n_in.x*sizeof(float), cudaMemcpyDeviceToHost);
    for (int weight = 0; weight < data.n_out.x*data.n_in.x; weight++) file << weights_vel[weight] << " ";
    delete[] weights_vel;
    file << "\n";

    file.close();
}

void fully_connected_layer::clear() {
    cudaFree(dev_weights);
    cudaFree(dev_weights_vel);
    cudaFree(dev_weights_updt);
    cudaFree(dev_biases);
    cudaFree(dev_biases_vel);
    cudaFree(dev_biases_updt);
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
                            z[get_data_index(map, y, x, data)] +=
                                    weights[get_convolutional_weights_index(previous_map, map, kernel_y, kernel_x, data)] *
                                    a[get_data_index(previous_map, y * data.stride_length + kernel_y, x * data.stride_length + kernel_x, data_previous)];
                        }
                    }
                }
                z[get_data_index(map, y, x, data)] += biases[map];
                new_a[get_data_index(map, y, x, data)] = data.activationFunct(z[get_data_index(map, y, x, data)]);
                new_dz[get_data_index(map, y, x, data)] = data.activationFunctPrime(z[get_data_index(map, y, x, data)]);
            }
        }
    }

    delete[] z;
}

void convolutional_layer::backprop(float * &delta,
                                   float* &activations,
                                   float* &derivative_z, float * &new_delta) {

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) delta[get_data_index(map, y, x, data)] *= derivative_z[get_data_index(map, y, x, data)];
        }
    }

    for (int map = 0; map < data.n_out.feature_maps; map++) {
        for (int y = 0; y < data.n_out.y; y++) {
            for (int x = 0; x < data.n_out.x; x++) {
                updateB[map] += delta[get_data_index(map, y, x, data)];
                for (int previous_map = 0; previous_map < data.n_in.feature_maps; previous_map++) {
                    for (int kernel_y = 0; kernel_y < data.receptive_field_length; kernel_y++) {
                        for (int kernel_x = 0; kernel_x < data.receptive_field_length; kernel_x++) {
                            new_delta[get_data_index(previous_map, y * data.stride_length + kernel_y, x * data.stride_length +
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

void max_pooling_layer::backprop(float * &delta,
                                 float* &activations, float* &derivative_z, float * &new_delta) {
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
                            new_delta[get_data_index(map, y * data.summarized_region_length + kernel_y,
                                    x * data.summarized_region_length + kernel_x, data_previous)] = delta[get_data_index(map, y, x, data)];
                        }
                    }
                }
            }
        }
    }
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

void input_layer::backprop(float * &delta,
                           float* &activations, float* &derivative_z, float * &new_delta) {
    (void) delta;
    (void) activations;
    (void) derivative_z;
    (void) new_delta;
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