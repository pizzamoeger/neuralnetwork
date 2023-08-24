#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

struct hyperparams {
    float convolutional_weights_learning_rate;
    float convolutional_biases_learning_rate;
    float fully_connected_weights_learning_rate;
    float fully_connected_biases_learning_rate;
    float convWRed;
    float convBRed;
    float fcWRed;
    float fcBRed;
    float L2_regularization_term;
    float momentum_coefficient;
    int epochs;
    int mini_batch_size;
    int training_data_size;
    int test_data_size;
    int cost;
};

enum {
    LAYER_NUM_FULLY_CONNECTED,
    LAYER_NUM_CONVOLUTIONAL,
    LAYER_NUM_MAX_POOLING,
    LAYER_NUM_INPUT
};

enum {
    SIGMOID,
    RELU,
    SOFTMAX
};

enum {
    CROSSENTROPY,
    MSE
};

#define OUTPUT_NEURONS 10
#define INPUT_NEURONS 28*28

__constant__ int zero = 0;
extern int* zero_pointer;
extern float* f_zero_pointer;

#define inline __noinline__ // TODO: decide if we should use this or not-> if not change if statements in activationfunc,... to switch

inline __device__ float activation_function(float x, int activation_func, float sum_of_exp);
inline __device__ float activation_function_prime(float x, int activation_func, float sum_of_exp);

pair<vector<pair<float*,float*>>, int> load_data(string filename); // TODO : watch this https://www.youtube.com/watch?v=m7E9piHcfr4 to make this faster
hyperparams get_params();
void clear_data(vector<pair<float*,float*>> & data);

struct network_data {
    int x;
    int y;
    int feature_maps;
};

struct layer_data {
    int type;

    network_data n_in;
    network_data n_out;

    int elems;

    bool last_layer = false;

    int stride_length;
    int receptive_field_length;

    int summarized_region_length;

    int activation_function;
};

struct layer {
    layer_data data;
    layer_data* dev_data;
    virtual void init(layer_data data, layer_data data_previous) = 0;
    virtual void feedforward(float* &a, float* &dz, float* &dev_z) = 0;
    virtual void backprop(float* &delta, float* &activations, float* &derivative_z, int* elems) = 0;
    virtual void update(hyperparams* params) = 0;
    virtual void save(string file) = 0;
    virtual void clear() = 0;
};

struct fully_connected_layer : public layer {
    layer_data* dev_data_previous;

    // biases[i] is bias of ith neuron.
    float* dev_biases;
    float* dev_biases_vel;
    float* dev_biases_updt;

    // weights[i][j] is weight of ith neuron to jth neuron in previous layer.
    float* dev_weights;
    float* dev_weights_vel;
    float* dev_weights_updt;

    void init (layer_data data, layer_data data_previous);

    void feedforward(float* &a, float* &dz, float* &z);

    void backprop(float* & delta, float* &activations, float* &derivative_z, int* elems);

    void update(hyperparams* params);

    void save(string filename);

    void clear();
};

/*struct convolutional_layer : public layer {

    layer_data data_previous;

    // biases[i] is bias of ith neuron.
    float* biases;
    float* biasesVelocity;

    // weights[i][j] is weight of ith neuron to jth neuron in previous layer.
    float* weights;
    float* weightsVelocity;

    // what needs to be updated
    float* updateB;
    float* updateW;

    int weights_size;

    void init (layer_data data, layer_data data_previous);

    void feedforward(float* a, float* dz, float* &new_a, float* &new_dz);

    void backprop(float* &delta, float* &activations, float* &derivative_z, float* &new_delta);

    void update(hyperparams params);

    void save(string filename);

    void clear();
};

struct max_pooling_layer : public layer {

    //layer_data data;
    layer_data data_previous;

    // no biases or velocities

    void init (layer_data data, layer_data data_previous);

    void feedforward(float* a, float* dz, float* &new_a, float* &new_dz);

    void backprop(float* &delta, float* &activations, float* &derivative_z, float* &new_delta);

    void update(hyperparams params);

    void save(string filename);

    void clear();
};*/

struct input_layer : public layer {

    //layer_data data;
    // no biases or velocities

    void init (layer_data data, layer_data data_previous);

    void feedforward(float* &a, float* &dz, float* &dev_z);

    void backprop(float* &delta, float* &activations, float* &derivative_z, int* elems);

    void update(hyperparams* params);

    void save(string filename);

    void clear();
};

struct Network {
    // number of layers
    int L;

    // layers
    unique_ptr<layer> *layers;

    hyperparams params;
    hyperparams* dev_params;

    void init (layer_data* layers, int L, hyperparams params);

    pair<float*, float*> feedforward(float* a);

    void SGD(vector<pair<float*,float*>> training_data, vector<pair<float*,float*>> test_data);

    void update_mini_batch(vector<pair<float*,float*>> mini_batch);

    void backprop(float* in, float* out);

    void save(string filename);

    void load(string filename);

    void clear();

    pair<int,int> evaluate(vector<pair<float*,float*>> test_data, int test_data_size);
};

int get_convolutional_weights_index(int previous_map, int map, int y, int x, layer_data &data);
int get_data_index(int map, int y, int x, layer_data &data);
inline __device__ int get_fully_connected_weight_index_dev (int neuron, int previous_neuron, int data_n_in);

__global__ void calc_a_and_dz (float* z, float* new_a, float* new_dz, int* af, float* sum_of_exp);
__global__ void set_delta (float* delta, float* activations, float* out, int* cost_func);
__global__ void backprop_logic (float* dev_weights_upt, float* dev_delta, float* dev_activations, float* dev_new_delta, float* dev_weights, int* data_n_in_x);  // TODO: faster with https://cuvilib.com/Reduction.pdf
__global__ void update_bias_vel (float* biases_vel, float* biases_updt, hyperparams* params);
__global__ void update_weights_vel (float* weights_vel, float* weights_updt, hyperparams* params);
__global__ void update_weights (float* weights, float* weights_vel, hyperparams* params);

__global__ void set_to (float *vec, float value); // initialize the elements to value
__global__ void set_to_random (float *vec, float* stddev); // initialize the elements to random value with mean 0 and given stddev
__global__ void add (float *vec_a, float *vec_b); // vec_a += vec_b
__global__ void mult (float *vec_a, float *vec_b); // vec_a[i] *= vec_b[i+offset_b]
__global__ void calc_exp (float* res, float* vec, int* max_id); // sum += exp(vec[i+offset])
__global__ void find_max (float* vec, int* id, int* size); // id is the id of the max elem in vec  // TODO: faster with https://cuvilib.com/Reduction.pdf

// TODO: remove offsets because i don't need them idk i am stupid

inline __device__ void reduce_last_warp(volatile float* sum, int ind, int block_size);
__global__ void reduce(float* input, float* res, int* size, int* block_size_ptr);
// https://stackoverflow.com/questions/29906486/cuda-multiple-parallel-reductions-sometimes-fail
/*
 * Suppose for example, that the input data has exactly 32 elements - the number of threads in a warp. In such scenario a single warp can be assigned to perform the reduction. Given that warp executes in a perfect sync, many __syncthreads() instructions can be removed - when compared to a block-level reduction.
 *
 * prev_map*prev_y*prev_x is max dimension of previous. has to be <= 1024 so that i can use it in one block (for parallel reduction).
 * not possible unless we only have one map. split up into multiple maps
 * blockIdx = for which previous_map
 * blockIdy = for which map
 * blcokIdz = for which x and y ?
 * */

#endif