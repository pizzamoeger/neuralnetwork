#include "includes.h"

std::random_device rd;
std::default_random_engine generator(rd());

void fully_connected_layer::init(layer_data data, layer_data data_previous, float* new_delta) {

    data.n_in = {data_previous.n_out.feature_maps * data_previous.n_out.y * data_previous.n_out.x, 1, 1};
    data.elems = data.n_in.x+data_previous.elems;
    this->data = data;

    cudaMalloc((void**) &delta, data.n_out.x*sizeof(float));
    this->new_delta = new_delta;

    cudaMalloc((void**) &this->dev_data, sizeof(layer_data));
    cudaMalloc((void**) &this->dev_data_previous, sizeof(layer_data));

    cudaMemcpy(this->dev_data, &data, sizeof(layer_data), cudaMemcpyHostToDevice);
    cudaMemcpy(this->dev_data_previous, &data_previous, sizeof(layer_data), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &dev_weights, data.n_out.x*data.n_in.x*sizeof(float));
    cudaMalloc((void**) &dev_weights_vel, data.n_out.x*data.n_in.x*sizeof(float));
    cudaMalloc((void**) &dev_weights_updt, data.n_out.x*data.n_in.x*sizeof(float));

    // weights init: https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/
    // https://wandb.ai/sauravmaheshkar/initialization/reports/A-Gentle-Introduction-To-Weight-Initialization-for-Neural-Networks--Vmlldzo2ODExMTg
    // https://stats.stackexchange.com/questions/373136/softmax-weights-initialization
    float stddev;
    if (data.activation_function == RELU) stddev = sqrt(2.0/data.n_in.x); // He-et-al
    else stddev = sqrt(2.0/data.n_in.x+data.n_in.x); // Xavier
    float* dev_stddev;
    cudaMalloc((void**) &dev_stddev, sizeof(float));
    cudaMemcpy(dev_stddev, &stddev, sizeof(float), cudaMemcpyHostToDevice);
    set_to_random<<<data.n_out.x * data.n_in.x, 1>>>(dev_weights, dev_stddev);
    set_to<<<data.n_out.x * data.n_in.x, 1>>>(dev_weights_vel, 0);
    set_to<<<data.n_out.x * data.n_in.x, 1>>>(dev_weights_updt, 0);

    cudaMalloc((void**) &dev_biases, data.n_out.x*sizeof(float));
    cudaMalloc((void**) &dev_biases_vel, data.n_out.x*sizeof(float));
    cudaMalloc((void**) &dev_biases_updt, data.n_out.x*sizeof(float));
    // biases init: https://medium.com/@glenmeyerowitz/bias-initialization-in-a-neural-network-2e5d26fed0f0
    set_to<<<data.n_out.x, 1>>>(dev_biases, 0.01);
    set_to<<<data.n_out.x,1>>>(dev_biases_vel, 0);
    set_to<<<data.n_out.x,1>>>(dev_biases_updt, 0);

    cudaFree(dev_stddev);
}

void fully_connected_layer::feedforward(float* dev_a, float* dev_dz) {
/*
    if (data.activation_function == SOFTMAX) {
        // TODO: make this smart
        reduce<<<data.n_out.x, data.n_in.x, data.n_in.x*sizeof(float)>>>(dev_weights, &dev_a[data.elems], &dev_data->n_in.x, &dev_data->n_in.x, CALC_Z, &dev_a[data.elems-data.n_in.x], dev_biases);
        cudaDeviceSynchronize();

        float* exp_vec;
        float* sum_of_exp;
        cudaMalloc((void**) &exp_vec, data.n_out.x*sizeof(float));
        cudaMalloc((void**) &sum_of_exp, sizeof(float));
        set_to<<<1,1>>> (sum_of_exp, 0);
        cudaDeviceSynchronize();
        //assert(data.n_out.x < (1<<10));

        int *max_id;
        cudaMalloc((void**) &max_id, sizeof(int));
        find_max<<<1,1>>>(&dev_a[data.elems], max_id, &dev_data->n_out.x);
        calc_exp<<<data.n_out.x, 1>>>(exp_vec, &dev_a[data.elems], max_id); // this could also be done in the reduce func
        cudaDeviceSynchronize();

        reduce<<<1, data.n_out.x, data.n_out.x*sizeof(float)>>>(exp_vec, sum_of_exp, &dev_data->n_out.x, &dev_data->n_out.x, ADD_EXP);
        cudaDeviceSynchronize();

        calc_a_and_dz<<<data.n_out.x, 1>>>(&dev_a[data.elems], &dev_dz[data.elems], &dev_data->activation_function, sum_of_exp);
        cudaDeviceSynchronize();

        cudaFree(max_id);
        cudaFree(exp_vec);
        cudaFree(sum_of_exp);
        //cudaDeviceSynchronize();
    } else {*/
        //reduce<<<data.n_out.x, data.n_in.x, data.n_in.x*sizeof(float)>>>(dev_weights, &dev_a[data.elems], &dev_data->n_in.x, &dev_data->n_in.x, CALC_Z, &dev_a[data.elems-data.n_in.x], dev_biases, &dev_dz[data.elems], &dev_data->activation_function
        reduce<<<data.n_out.x, (1<<10), data.n_in.x*sizeof(float)>>>(dev_weights, &dev_a[data.elems], &dev_data->n_in.x, &dev_data->n_in.x, CALC_Z, &dev_a[data.elems-data.n_in.x], dev_biases, &dev_dz[data.elems], &dev_data->activation_function);
    cudaDeviceSynchronize();
    //}
}

void fully_connected_layer::backprop(float* activations, float* derivative_z) {
    // TODO: this could be made faster but also uglier
    backprop_logic<<<data.n_out.x,data.n_in.x>>>(dev_weights_updt, delta, &activations[data.elems-data.n_in.x], dev_biases_updt, &dev_data->n_in.x);
    reduce<<<data.n_in.x, (1<<10), data.n_out.x*sizeof(float)>>>(dev_weights, new_delta, &dev_data->n_out.x, &dev_data->n_out.x, CALC_ND, delta, &derivative_z[data.elems-data.n_in.x]);
    cudaDeviceSynchronize();
}

void fully_connected_layer::update(hyperparams* dev_params) {
    // update velocities
    ::update<<<data.n_out.x, data.n_in.x>>> (dev_biases_vel, dev_weights_vel, dev_weights_updt, dev_biases_updt, dev_weights, dev_biases, dev_params);
    cudaDeviceSynchronize();
}

void fully_connected_layer::save(std::string filename) {
    std::ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_FULLY_CONNECTED << "//";
    file << data.activation_function << "//";
    file << data.n_out.x << "//";

    float* biases = new float [data.n_out.x];
    cudaMemcpy(biases, dev_biases, data.n_out.x*sizeof(float), cudaMemcpyDeviceToHost);
    for (int bias = 0; bias < data.n_out.x; bias++) file << biases[bias] << " ";
    delete[] biases;
    file << "//";

    float* biases_vel = new float [data.n_out.x];
    cudaMemcpy(biases_vel, dev_biases_vel, data.n_out.x*sizeof(float), cudaMemcpyDeviceToHost);
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
    cudaFree(delta);
    cudaFree(dev_weights);
    cudaFree(dev_weights_vel);
    cudaFree(dev_weights_updt);
    cudaFree(dev_biases);
    cudaFree(dev_biases_vel);
    cudaFree(dev_biases_updt);
    cudaFree(dev_data_previous);
    cudaFree(dev_data);
}

/*void convolutional_layer::init(layer_data data, layer_data data_previous) {

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
                new_a[get_data_index(map, y, x, data)] = activation_function(z[get_data_index(map, y, x, data)], data.activation_function);
                new_dz[get_data_index(map, y, x, data)] = activation_function_prime(z[get_data_index(map, y, x, data)], data.activation_function);
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
    file << data.activation_function << "//";
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
*/
void input_layer::init(layer_data data, layer_data data_previous, float* new_delta) {
    data.elems = 0;
    this->data = data;
    cudaMalloc((void**) &delta, data.n_out.feature_maps*data.n_out.y*data.n_out.x*sizeof(float));
    (void) data_previous;
}

void input_layer::feedforward(float* a, float* dz) {}

void input_layer::backprop(float* activations, float* derivative_z) {}

void input_layer::update(hyperparams* params) {}

void input_layer::save(std::string filename) {
    std::ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_INPUT << "//";
    file << data.n_out.x << " " << data.n_out.y << "\n";

    file.close();
}

void input_layer::clear() {
    cudaFree(delta);
}
