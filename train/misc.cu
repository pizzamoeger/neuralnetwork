#include "includes.h"

// sigmoid function and its derivative
inline __device__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

inline __device__ float sigmoid_prime(float x) {
    return (sigmoid(x)*(1-sigmoid(x)));
}

inline __device__ float relu(float x){
    return max(x, 0.0f);
}

inline __device__ float relu_prime(float x){
    if (x > 0) return 1.0f;
    return 0.0f;
}

inline __device__ float softmax(float x, float sum_of_exp) {
    return expf(x)/sum_of_exp;
}

inline __device__ float softmax_prime(float x, float sum_of_exp) {
    return (softmax(x, sum_of_exp)*(1-softmax(x, sum_of_exp)));
}

inline __device__ float cross_entropy_prime(float out_net, float out_cor) {
    return (out_net-out_cor);
}

inline __device__ float activation_function(float x, int activation_func, float sum_of_exp) {
    switch (activation_func) {
        case SIGMOID:
            return sigmoid(x);
        case RELU:
            return relu(x);
        case SOFTMAX:
            return softmax(x, sum_of_exp);
        default:
            return 0;
    }
}

inline __device__ float activation_function_prime(float x, int activation_func, float sum_of_exp) {
    switch (activation_func) {
        case SIGMOID:
            return sigmoid_prime(x);
        case RELU:
            return relu_prime(x);
        case SOFTMAX:
            return softmax_prime(x, sum_of_exp);
        default:
            return 0;
    }
}

inline __device__ float cost_function_prime(float out_net, float out_cor, int cost_function) {
    if (cost_function == CROSSENTROPY) return cross_entropy_prime(out_net, out_cor);
    else return 0;
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

inline __device__ int get_fully_connected_weight_index_dev (int neuron, int previous_neuron, int data_n_in) {
    return neuron*data_n_in+previous_neuron;
}

// load data
std::pair<std::vector<std::pair<float*,float*>>, int> load_data(std::string filename) {
    // loads data from csv file of form label, pixel1, pixel2, pixel3, ..., pixel784
    std::ifstream file;
    std::string line;

    file.open(filename);

    // how many lines there are in the file
    int dataPoints = 0;
    while (getline(file, line)) {
        dataPoints++;
    }

    file.clear(); // Reset stream state
    file.seekg(0); // Move cursor back to beginning

    int lineIndex = 0;
    std::vector<std::pair<float*,float*>> data (dataPoints, {nullptr, nullptr});

    while (getline(file, line)) {
        std::stringstream ss(line);
        float* data_in = new float [INPUT_NEURONS];
        float* data_out = new float [OUTPUT_NEURONS];

        for (int i = 0; i < INPUT_NEURONS; i++) data_in[i] = 0;
        for (int i = 0; i < OUTPUT_NEURONS; i++) data_out[i] = 0;

        int label = -1;
        int i = 0;
        while (ss.good()) {
            std::string substr;
            getline(ss, substr, ' ');
            if (label == -1) {
                label = stoi(substr);
            } else {
                if (i == INPUT_NEURONS) break;
                data_in[i] = atof(substr.c_str());
                i++;
            }
        }
        data_out[label] = 1;


        float* dev_data_in;
        float* dev_data_out;
        cudaMalloc((void**) &dev_data_in, INPUT_NEURONS*sizeof(float));
        cudaMalloc((void**) &dev_data_out, OUTPUT_NEURONS*sizeof(float));
        cudaMemcpy(dev_data_in, data_in, INPUT_NEURONS*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_data_out, data_out, OUTPUT_NEURONS*sizeof(float), cudaMemcpyHostToDevice);
        data[lineIndex] = {dev_data_in, dev_data_out};

        lineIndex++;

        delete [] data_in;
        delete [] data_out;
    }

    std::cerr << dataPoints << " data loaded from " + filename + "\n";
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

    params.cost = CROSSENTROPY;

    return params;
}

void clear_data(std::vector<std::pair<float*,float*>> & data) {
    for (int data_point = 0; data_point < (int)data.size(); data_point++) {
        cudaFree(data[data_point].first);
        cudaFree(data[data_point].second);
    }
}

__global__ void set_delta (float* delta, float* activations, float* out, int* cost_func) {
    int neuron = blockIdx.x;
    delta[neuron] = cost_function_prime(activations[neuron], out[neuron], *cost_func);
}

__global__ void backprop_logic (float* dev_weights_upt, float* dev_delta, float* dev_activations, float* dev_biases_updt, int* data_n_in_x) {
    int neuron = blockIdx.x;
    int previous_neuron = threadIdx.x;
    dev_weights_upt[get_fully_connected_weight_index_dev(neuron, previous_neuron, *data_n_in_x)] += dev_delta[neuron] * dev_activations[previous_neuron];
    if (previous_neuron == 0) dev_biases_updt[neuron] += dev_delta[neuron];
}

__global__ void update (float* biases_vel, float* weights_vel, float* weights_updt, float* biases_updt, float* weights, float* biases, hyperparams* params) {
    int neuron = blockIdx.x;
    int previous_neuron = threadIdx.x;
    int weight = neuron*blockDim.x+previous_neuron;

    if (previous_neuron == 0) {
        biases_vel[neuron] = params->momentum_coefficient * biases_vel[neuron] -
                                 (params->fully_connected_biases_learning_rate / params->mini_batch_size) *
                                 biases_updt[neuron];
        biases[neuron] += biases_vel[neuron];
        biases_updt[neuron] = 0;
    }

    weights_vel[weight] =
            params->momentum_coefficient * weights_updt[weight] -
            (params->fully_connected_weights_learning_rate / params->mini_batch_size) *
            weights_updt[weight];

    weights[weight] = (1 - params->fully_connected_weights_learning_rate * params->L2_regularization_term
                           / params->training_data_size) * weights[weight] + weights_vel[weight];

    weights_updt[weight] = 0;
}

__global__ void eval (float* correct, float* output, int* counter, int* size) {
    int index = 0;

    for (int i = 0; i < (*size); i++) {
        if (output[i] > output[index]) index = i;
    }

    if (correct[index] == 1) (*counter)++;
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
    //printf("weightss: %f, %d\n", vec[index], index);
}

inline __device__ void reduce_last_warp(volatile float* sum, int ind, int block_size) {
    if (block_size > 32) {
        if (ind < block_size - 32 && ind < 32) sum[ind] += sum[ind + 32];
    }
    if (block_size > 16) {
        if (ind < block_size - 16 && ind < 16) sum[ind] += sum[ind + 16];
    }
    if (block_size > 8) {
        if (ind < block_size - 8 && ind < 8) sum[ind] += sum[ind + 8];
    }
    if (block_size > 4) {
        if (ind < block_size - 4 && ind < 4) sum[ind] += sum[ind + 4];
    }
    if (block_size > 2) {
        if (ind < block_size - 2 && ind < 2) sum[ind] += sum[ind + 2];
    }
    if (block_size > 1) {
        if (ind < block_size - 1 && ind < 1) sum[ind] += sum[ind + 1];
    }
}

inline __device__ float calc_input(int calc, int bid, int tid, int size, float* inpt, float* mult_n) {
    switch (calc) {
        case CALC_Z: {
            float ret = inpt[bid * size + tid] * mult_n[tid];
            return ret;
        }
        case CALC_ND: {
            float ret = inpt[tid * size + bid] * mult_n[tid];
            return ret;
        }
        case ADD_EXP:
            return inpt[bid*size+tid];
        default:
            return 0;
    }
}

inline __device__ void calc_res(int calc, int bid, float* res_1, float* res_2, int *activation_func, float *sum_of_exp, float* add_once) {
    switch (calc) {
        case CALC_Z:
            while(*activation_func == SOFTMAX);
            res_2[bid] = activation_function_prime(res_1[bid], *activation_func, 0);
            res_1[bid] = activation_function(res_1[bid], *activation_func, 0);
            break;
        case CALC_ND:
            res_1[bid] = res_1[bid]*add_once[bid];
        default:
            break;
    }
}

__global__ void reduce(float* input, float* res_1, network_data* size_tot, int calc, float* mult_1, float* vec_2, float* res_2, int* activation_func, int* stride_length) {
    const int block_size = blockDim.x*blockDim.y*blockDim.z;
    extern __shared__ float sum[];
    int tid = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
    int bid = blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x;

    int add = 0;
    int s = size_tot->x*size_tot->y*size_tot->feature_maps;
    if (calc == CALC_ND) s = gridDim.x;

    sum[tid] = 0;
    while (tid + add < size_tot->x*size_tot->y*size_tot->feature_maps) {
        if (stride_length != NULL) {
            // TODO: make this nicer
            int tid_a = threadIdx.z*size_tot->x*size_tot->y
                    + (blockIdx.y*(*stride_length)+threadIdx.y)*size_tot->x
                    + (blockIdx.x*(*stride_length)+threadIdx.x);

            sum[tid] += mult_1[tid_a]*input[blockIdx.z*block_size + tid];
            // idk make this better too
            break;
        } else sum[tid] += calc_input(calc, bid, tid+add, s, input, mult_1);
        add += block_size;
    }
    __syncthreads();

    if (block_size > 512) {
        if (tid < block_size - 512) sum[tid] += sum[tid + 512];
        __syncthreads();
    }
    if (block_size > 256) {
        if (tid < block_size - 256 && tid < 256) sum[tid] += sum[tid + 256];
        __syncthreads();
    }
    if (block_size > 128) {
        if (tid < block_size - 128 && tid < 128) sum[tid] += sum[tid + 128];
        __syncthreads();
    }
    if (block_size > 64) {
        if (tid < block_size - 64 && tid < 64) sum[tid] += sum[tid + 64];
        __syncthreads();
    }

    if (tid < 32) reduce_last_warp(sum, tid, block_size);
    if (tid == 0) {
        int bid_a = bid;
        if (stride_length != NULL) bid_a = blockIdx.z;
        if (calc == CALC_Z) res_1[bid] = sum[tid]+vec_2[bid_a];
        else res_1[bid] = sum[tid];
        calc_res(calc, bid, res_1, res_2, activation_func, NULL, vec_2);
    }
}