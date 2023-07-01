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
    layer_data data;
    virtual void init(layer_data data, layer_data data_previous) = 0;
    virtual void feedforward(float* a, float* dz, float* &new_a, float* &new_dz) = 0;
    virtual void backprop(vector<float> &delta, float* &activations, float* &derivative_z) = 0;
    virtual void update(hyperparams params) = 0;
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

    void init (layer_data data, layer_data data_previous);

    void feedforward(float* a, float* dz, float* &new_a, float* &new_dz);

    void backprop(vector<float> & delta, float* &activations, float* &derivative_z);

    void update(hyperparams params);

    void save(string filename);

    void clear();
};

struct convolutional_layer : public layer {

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

    void backprop(vector<float> &delta, float* &activations, float* &derivative_z);

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

    void backprop(vector<float> &delta, float* &activations, float* &derivative_z);

    void update(hyperparams params);

    void save(string filename);

    void clear();
};

struct input_layer : public layer {

    //layer_data data;
    // no biases or velocities

    void init (layer_data data, layer_data data_previous);

    void feedforward(float* a, float* dz, float* &new_a, float* &new_dz);

    void backprop(vector<float> &delta, float* &activations, float* &derivative_z);

    void update(hyperparams params);

    void save(string filename);

    void clear();
};

struct Network {
    // number of layers
    int L;

    // layers
    unique_ptr<layer> *layers;

    function<float(float, float)> costFunctPrime;

    void init (layer_data* layers, int L, function<float(float, float)> costFunctPrime, hyperparams params);

    pair<float**, float**> feedforward(input_type &a);

    void SGD(data_point* training_data, data_point* test_data, hyperparams params);

    void update_mini_batch(data_point* mini_batch, hyperparams params);

    void backprop(input_type &in, output_type &out);

    void save(string filename);

    void load(string filename);

    void clear();

    pair<int,int> evaluate(data_point* test_data, int test_data_size);
};