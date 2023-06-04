struct hyperparams {
    float convolutional_weights_learning_rate;
    float convolutional_biases_learning_rate;
    float fully_connected_weights_learning_rate;
    float fully_connected_biases_learning_rate;
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

float crossEntropyPrime(float output_activation, float y);
vector<pair<vector<float>, vector<float>>> load_data(string filename);
hyperparams get_params();

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
    virtual void feedforward(vector<float> &a, vector<float> &derivative_z) = 0;
    virtual void backprop(vector<float> &delta, vector<float> &activations, vector<float> &derivative_z) = 0;
    virtual void update(hyperparams params) = 0;
    virtual void save(string file) = 0;
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
    vector<float> biases;
    vector<float> biasesVelocity;

    // weights[i][j] is weight of ith neuron to jth neuron in previous layer.
    vector<float> weights;
    vector<float> weightsVelocity;

    // what needs to be updated
    vector<float> updateB;
    vector<float> updateW;

    void init (layer_data data, layer_data data_previous);

    void feedforward(vector<float> & a, vector<float> & derivative_z);

    void backprop(vector<float> & delta, vector<float> &activations, vector<float> &derivative_z);

    void update(hyperparams params);

    void save(string filename);

};

struct convolutional_layer : public layer {

    layer_data data_previous;

    // biases[i] is bias of ith neuron.
    vector<float> biases;
    vector<float> biasesVelocity;

    // weights[i][j] is weight of ith neuron to jth neuron in previous layer.
    vector<float> weights;
    vector<float> weightsVelocity;

    // what needs to be updated
    vector<float> updateB;
    vector<float> updateW;

    int weights_size;

    void init (layer_data data, layer_data data_previous);

    void feedforward(vector<float> &a, vector<float> &derivative_z);

    void backprop(vector<float> &delta, vector<float> &activations, vector<float> &derivative_z);

    void update(hyperparams params);

    void save(string filename);
};

struct max_pooling_layer : public layer {

    //layer_data data;
    layer_data data_previous;

    // no biases or velocities

    void init (layer_data data, layer_data data_previous);

    void feedforward(vector<float> &a, vector<float> &derivative_z);

    void backprop(vector<float> &delta, vector<float> &activations, vector<float> &derivative_z);

    void update(hyperparams params);

    void save(string filename);
};

struct input_layer : public layer {

    //layer_data data;
    // no biases or velocities

    void init (layer_data data, layer_data data_previous);

    void feedforward(vector<float> &a, vector<float> &derivative_z);

    void backprop(vector<float> &delta, vector<float> &activations, vector<float> &derivative_z);

    void update(hyperparams params);

    void save(string filename);
};

struct Network {
    // number of layers
    int L;

    // layers
    vector<unique_ptr<layer>> layers;

    function<float(float, float)> costFunctPrime;

    void init (vector<layer_data> & layers, function<float(float, float)> costFunctPrime);

    pair<vector<vector<float>>, vector<vector<float>>> feedforward(vector<float> &a);

    void SGD(vector<pair<vector<float>, vector<float>>> training_data, vector<pair<vector<float>, vector<float>>> test_data, hyperparams params);

    void update_mini_batch(vector<pair<vector<float>, vector<float>>> &mini_batch, hyperparams params);

    void backprop(vector<float> &in, vector<float> &out);

    void save(string filename);

    void load(string filename);

    pair<int,int> evaluate(vector<pair<vector<float>, vector<float>>> test_data, hyperparams params);
};