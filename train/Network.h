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
};

float sigmoid(float x);
float sigmoidPrime(float x);
float relu(float x);
float reluPrime(float x);

typedef float input_type[28*28];
typedef float output_type[10];
typedef pair<input_type, output_type> data_point;

float crossEntropyPrime(float output_activation, float y);
pair<data_point*, int> load_data(string filename);
hyperparams get_params();
void clear_data(data_point *data);

struct network_data {
    int x;
    int y;
    int feature_maps;
};

struct layer_data {
    int type;

    network_data n_in;
    network_data n_out;

    bool last_layer = false;

    int stride_length;
    int receptive_field_length;

    int summarized_region_length;

    function<float(float)> activationFunct;
    function<float(float)> activationFunctPrime;

};

struct layer {
    layer_data* dev_data;
    layer_data data;
    virtual void init(layer_data data, layer_data data_previous) = 0;
    virtual void feedforward(float* &a, float* &dz, float* &dev_z, int* elems) = 0;
    virtual void backprop(float* &delta, float* &activations, float* &derivative_z, int* elems) = 0;
    virtual void update(hyperparams* params) = 0;
    virtual void save(string file) = 0;
    virtual void clear() = 0;
};

enum {
    LAYER_NUM_FULLY_CONNECTED,
    LAYER_NUM_CONVOLUTIONAL,
    LAYER_NUM_MAX_POOLING,
    LAYER_NUM_INPUT
};

// TODO activation functions: a number which act func, make a function which you can call where it automatically calls the correct act func -> act func can be stored

struct fully_connected_layer : public layer {
    layer_data* dev_data_previous;
    layer_data data_previous;

    // biases[i] is bias of ith neuron.
    float* dev_biases;
    float* dev_biases_vel;
    float* dev_biases_updt;


    // weights[i][j] is weight of ith neuron to jth neuron in previous layer.
    float* dev_weights;
    float* dev_weights_vel;
    float* dev_weights_updt;

    void init (layer_data data, layer_data data_previous);

    void feedforward(float* &a, float* &dz, float* &z, int* elems);

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

    void feedforward(float* &a, float* &dz, float* &dev_z, int* elems);

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

    function<float(float, float)> costFunctPrime;

    void init (layer_data* layers, int L, function<float(float, float)> costFunctPrime);

    pair<float*, float*> feedforward(input_type &a);

    void SGD(data_point* training_data, data_point* test_data, hyperparams params);

    void update_mini_batch(data_point* mini_batch, hyperparams params, hyperparams* dev_params);

    void backprop(input_type &in, output_type &out);

    void save(string filename);

    void load(string filename);

    void clear();

    pair<int,int> evaluate(data_point* test_data, int test_data_size);
};

int get_convolutional_weights_index(int previous_map, int map, int y, int x, layer_data &data);
int get_data_index(int map, int y, int x, layer_data &data);
int get_fully_connected_weight_index(int neuron, int previous_neuron, int data_n_in);
__device__ int get_fully_connected_weight_index_dev (int neuron, int previous_neuron, int data_n_in);

__global__ void addWeights (float* a, float* weights, float* z, int* data_n_in, int* elems);
__global__ void getNewA (float* z, float* biases, float* new_a, float* new_dz, int* elems);
__global__ void backprop_logic (float* dev_weights_upt, float* dev_delta, float* dev_activations, float* dev_new_delta, float* dev_weights, int* data_n_in_x, int* offset);
__global__ void update_bias_vel (float* biases_vel, float* biases_updt, hyperparams* params);
__global__ void update_weights_vel (float* weights_vel, float* weights_updt, hyperparams* params);
__global__ void update_weights (float* weights, float* weights_vel, hyperparams* params);

__global__ void set_to (float *vec, float value); // initialize the elements to value
__global__ void set_to_random (float *vec, int* data_n_in_x); // initialize the elements to random value with mean 0 and stddev 1/sqrt(data_n_in_x
__global__ void add (float *vec_a, float *vec_b); // vec_a += vec_b
__global__ void mult (float *vec_a, float *vec_b, int* offset_b); // vec_a[i] *= vec_b[i+offset_b]