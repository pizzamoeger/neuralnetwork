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
float crossEntropyPrime(float output_activation, float y);
vector<pair<vector<vector<float>>, vector<float>>> load_data(string filename);
hyperparams get_params();

struct network_data {
    int x;
    int y;
};

struct layer_data {
    int type; // 0: input, 1: convolutional, 2: max pooling, 3: flatten, 4: fully connected

    network_data n_in;
    network_data n_out;

    int stride_length;
    int receptive_field_length;

    int feature_maps;
    int previous_feature_maps;

    int summarized_region_length;

    function<float(float)> activationFunct;
    function<float(float)> activationFunctPrime;

};

struct layer {
    virtual void init(layer_data data) = 0;
    virtual void feedforward(vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &derivative_z) = 0;
    virtual void backprop(vector<vector<vector<float>>> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &derivative_z) = 0;
    virtual void update(hyperparams params) = 0;
};

struct fully_connected_layer : public layer {
    layer_data data;

    // biases[i] is bias of ith neuron.
    vector<float> biases;
    vector<float> biasesVelocity;

    // weights[i][j] is weight of ith neuron to jth neuron in previous layer.
    vector<vector<float>> weights;
    vector<vector<float>> weightsVelocity;

    // what needs to be updated
    vector<float> updateB;
    vector<vector<float>> updateW;

    void init (layer_data data);

    void feedforward(vector<vector<vector<float>>> & a, vector<vector<vector<float>>> & derivative_z);

    void backprop(vector<vector<vector<float>>> & delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &derivative_z);

    void update(hyperparams params);

};

struct convolutional_layer : public layer {

    layer_data data;

    // biases[i] is bias of ith neuron.
    vector<float> biases;
    vector<float> biasesVelocity;

    // weights[i][j] is weight of ith neuron to jth neuron in previous layer.
    vector<vector<vector<vector<float>>>> weights;
    vector<vector<vector<vector<float>>>> weightsVelocity;

    // what needs to be updated
    vector<float> updateB;
    vector<vector<vector<vector<float>>>> updateW;

    void init (layer_data data);

    void feedforward(vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &derivative_z);

    void backprop(vector<vector<vector<float>>> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &derivative_z);

    void update(hyperparams params);

};

struct max_pooling_layer : public layer {

    layer_data data;

    // no biases or velocities

    void init (layer_data data);

    void feedforward(vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &derivative_z);

    void backprop(vector<vector<vector<float>>> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &derivative_z);

    void update(hyperparams params);

};

struct flatten_layer : public layer {

    layer_data data;

    // no biases or velocities

    void init (layer_data data);

    void feedforward(vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &derivative_z);

    void backprop(vector<vector<vector<float>>> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &derivative_z);

    void update(hyperparams params);

};

struct input_layer : public layer {

    layer_data data;
    // no biases or velocities

    void init (layer_data data);

    void feedforward(vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &derivative_z);

    void backprop(vector<vector<vector<float>>> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &derivative_z);

    void update(hyperparams params);

};

struct Network {
    // number of layers
    int L;

    // layers
    vector<unique_ptr<layer>> layers;

    function<float(float, float)> costFunctPrime;

    void init (vector<layer_data> & layers, function<float(float, float)> costFunctPrime);

    pair<vector<vector<vector<vector<float>>>>, vector<vector<vector<vector<float>>>>> feedforward(vector<vector<float>> &a);

    void SGD(vector<pair<vector<vector<float>>, vector<float>>> training_data, vector<pair<vector<vector<float>>, vector<float>>> test_data, hyperparams params);

    void update_mini_batch(vector<pair<vector<vector<float>>, vector<float>>> &mini_batch, hyperparams params);

    void backprop(vector<vector<float>> &in, vector<float> &out);

    void save(string filename);

    void load(string filename);
};