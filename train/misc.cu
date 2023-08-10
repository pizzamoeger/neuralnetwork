#include "includes.h"

// sigmoid function and its derivative
__device__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}
__device__ float sigmoid_prime(float x) {
    return (sigmoid(x)*(1-sigmoid(x)));
}

__device__ float relu(float x){
    return max(x, 0.0f);
}

__device__ float relu_prime(float x){
    if (x > 0) return 1.0f;
    return 0.0f;
}

__device__ float softmax(float x, float sum_of_exp) {
    return expf(x)/sum_of_exp;
}

__device__ float softmax_prime(float x, float sum_of_exp) {
    return (softmax(x, sum_of_exp)*(1-softmax(x, sum_of_exp)));
}
__device__ float activation_function(float x, int activation_func, float sum_of_exp) {
    switch (activation_func) {
        case SIGMOID:
            return sigmoid(x);
        case RELU:
            return relu(x);
        case SOFTMAX:
            return softmax(x, sum_of_exp);
    }
    return 0;
}

__device__ float activation_function_prime(float x, int activation_func, float sum_of_exp) {
    switch (activation_func) {
        case SIGMOID:
            return sigmoid_prime(x);
        case RELU:
            return relu_prime(x);
        case SOFTMAX:
            return softmax_prime(x, sum_of_exp);
    }
    return 0;
}

// cross entropy cost function
float crossEntropyPrime(float output_activation, float y) {
    return (output_activation-y);
}

int get_convolutional_weights_index(int previous_map, int map, int y, int x, layer_data &data) {
    return
            previous_map * (data.n_out.feature_maps * data.receptive_field_length * data.receptive_field_length)
            + map * (data.receptive_field_length * data.receptive_field_length)
            + y * (data.receptive_field_length)
            + x;
}

int get_data_index(int map, int y, int x, layer_data &data) {
    return
            map * (data.n_out.x * data.n_out.y)
            + y * (data.n_out.x)
            + x;
}

int get_fully_connected_weight_index(int neuron, int previous_neuron, int data_n_in) {
    return neuron*data_n_in+previous_neuron;
}

__device__ int get_fully_connected_weight_index_dev (int neuron, int previous_neuron, int data_n_in) {
    return neuron*data_n_in+previous_neuron;
}

// load data
pair<data_point*, int> load_data(string filename) {
    // loads data from csv file of form label, pixel1, pixel2, pixel3, ..., pixel784
    ifstream file;
    string line;

    file.open(filename);

    // how many lines there are in the file
    int dataPoints = 0;
    while (getline(file, line)) {
        dataPoints++;
    }
    file.close();

    file.open(filename);

    data_point *data = new data_point[dataPoints];
    int lineIndex = 0;

    while (getline(file, line)) {
        stringstream ss(line);

        for (int i = 0; i < 10; i++) data[lineIndex].second[i] = 0;
        for (int i = 0; i < 28 * 28; i++) data[lineIndex].first[i] = 0;

        int label = -1;
        int i = 0;
        while (ss.good()) {
            string substr;
            getline(ss, substr, ' ');
            if (label == -1) {
                label = stoi(substr);
            } else {
                if (i == 28 * 28) break;
                data[lineIndex].first[i] = atof(substr.c_str());
                i++;
            }
        }
        data[lineIndex].second[label] = 1;
        lineIndex++;
    }
    cerr << dataPoints << " data loaded from " + filename + "\n";
    file.close();
    return {data, dataPoints};
}

hyperparams get_params() {
    hyperparams params;

    params.mini_batch_size = 16;
    params.epochs = 5;

    params.fully_connected_weights_learning_rate = 1.2*0.017599067515299563;
    params.fully_connected_biases_learning_rate = 1.2*0.041000786959874205;
    params.convolutional_weights_learning_rate = 1.2*1.0075;
    params.convolutional_biases_learning_rate = 1.2*0.011;

    params.L2_regularization_term = 0;
    params.momentum_coefficient = 0;

    return params;
}

void clear_data(data_point *data) {
    delete[] data;
}

__global__ void calc_z (float* a, float* weights, float* biases, float* z, int* data_n_in, int* offset) {
    int neuron = blockIdx.x;
    int previous_neuron = threadIdx.x;
    if (previous_neuron == 0) atomicAdd(&z[(*offset)+neuron], biases[neuron]);
    atomicAdd(&z[(*offset)+neuron], weights[get_fully_connected_weight_index_dev(neuron, previous_neuron, *data_n_in)] * a[(*offset)-(*data_n_in)+previous_neuron]);
}

__global__ void calc_a_and_dz (float* z, float* new_a, float* new_dz, int* offset, int* activation_func, float* sum_of_exp) {
    int neuron = blockIdx.x;

    new_a[(*offset)+neuron] = activation_function(z[(*offset) + neuron], *activation_func, *sum_of_exp);
    new_dz[(*offset)+neuron] = activation_function_prime(z[(*offset) + neuron], *activation_func, *sum_of_exp);
}

__global__ void backprop_logic (float* dev_weights_upt, float* dev_delta, float* dev_activations, float* dev_new_delta, float* dev_weights, int* data_n_in_x, int *offset) {
    int neuron = blockIdx.x;
    int previous_neuron = threadIdx.x;
    atomicAdd(&dev_weights_upt[get_fully_connected_weight_index_dev(neuron, previous_neuron, *data_n_in_x)], dev_delta[neuron] * dev_activations[(*offset)-(*data_n_in_x)+previous_neuron]);
    atomicAdd(&dev_new_delta[previous_neuron], dev_delta[neuron] * dev_weights[get_fully_connected_weight_index_dev(neuron, previous_neuron, *data_n_in_x)]);
}

__global__ void update_bias_vel (float* biases_vel, float* biases_updt, hyperparams* params) {
    int neuron = blockIdx.x;
    biases_vel[neuron] = params->momentum_coefficient * biases_vel[neuron] -
                             (params->fully_connected_biases_learning_rate / params->mini_batch_size) *
                             biases_updt[neuron];
}

__global__ void update_weights_vel (float* weights_vel, float* weights_updt, hyperparams* params) {
    int weight = blockIdx.x;
    weights_vel[weight] =
            params->momentum_coefficient * weights_updt[weight] -
            (params->fully_connected_weights_learning_rate / params->mini_batch_size) *
            weights_updt[weight];
}

__global__ void update_weights (float* weights, float* weights_vel, hyperparams* params) {
    int weight = blockIdx.x;
    weights[weight] = (1 - params->fully_connected_weights_learning_rate * params->L2_regularization_term
                        / params->training_data_size) * weights[weight] + weights_vel[weight];
}

__global__ void set_to (float *vec, float value) {
    int index = blockIdx.x;
    vec[index] = value;
}

__global__ void set_to_random (float *vec, float *stddev) {
    int index = blockIdx.x;

    curandState state;
    curand_init(clock64(), index, 0, &state);
    vec[index] = curand_normal(&state)*(*stddev);
}

__global__ void add (float *vec_a, float *vec_b) {
    int index = blockIdx.x;
    vec_a[index] += vec_b[index];
}

__global__ void mult (float *vec_a, float *vec_b, int *offset_b) {
    int index = blockIdx.x;
    vec_a[index] *= vec_b[index+(*offset_b)];
}

__global__ void calc_sum_of_exp (float* sum, float* vec, int* offset) {
    int index = blockIdx.x+(*offset);
    atomicAdd(sum, expf(vec[index]));
}