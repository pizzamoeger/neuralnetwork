struct hyperparams {
    double learning_rate;
    double L2_regularization_term;
    double momentum_coefficient;
    int epochs;
    int mini_batch_size;
    int training_data_size;
    int test_data_size;
};

double sigmoid(double x);
double sigmoidPrime(double x);
double crossEntropyPrime(double output_activation, double y);
vector<pair<vector<vector<double>>, vector<double>>> load_data(string filename);
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

    int summarized_region_length;
};

struct layer {
    int feature_maps = 1;
    virtual void init(layer_data data, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime) = 0;
    virtual void feedforward(int previous_feature_maps, vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &z) = 0;
    virtual void backprop(int previous_feature_maps, vector<float> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z) = 0;
    virtual void update(hyperparams params) = 0;
};

struct fully_connected_layer : public layer {
    // number of neurons
    int n_in;
    int n_out;

    // biases[i] is bias of ith neuron.
    vector<double> biases;
    vector<double> biasesVelocity;

    // weights[i][j] is weight of ith neuron to jth neuron in previous layer.
    vector<vector<double>> weights;
    vector<vector<double>> weightsVelocity;

    // what needs to be updated
    vector<double> updateB;
    vector<vector<double>> updateW;

    // activation function
    function<double(double)> activationFunct;
    function<double(double)> activationFunctPrime;

    // cost function
    function<double(double, double)> costFunctPrime;

    void init (layer_data data, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime);

    void feedforward(int _, vector<vector<vector<double>>> & a, vector<vector<vector<double>>> & z);

    void backprop(int _, vector<double> & delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z);

    void update(hyperparams params);

};

struct convolutional_layer : public layer {
    // number of neurons
    network_data n_in;
    network_data n_out;

    int stride_length;
    int receptive_field_length;

    // biases[i] is bias of ith neuron.
    vector<double> biases;
    vector<double> biasesVelocity;

    // weights[i][j] is weight of ith neuron to jth neuron in previous layer.
    vector<vector<vector<double>>> weights;
    vector<vector<vector<double>>> weightsVelocity;

    // what needs to be updated
    vector<double> updateB;
    vector<vector<vector<double>>> updateW;

    // activation function
    function<double(double)> activationFunct;
    function<double(double)> activationFunctPrime;

    // cost function
    function<double(double, double)> costFunctPrime;

    void init (layer_data data, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime);

    void feedforward(int previous_feature_maps, vector<vector<vector<double>>> &a, vector<vector<vector<double>>> &z);

    void backprop(int previous_feature_maps, vector<float> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z);

    void update(hyperparams params);

};

struct max_pooling_layer : public layer {
    // number of neurons
    network_data n_in;
    network_data n_out;

    int summarized_region_length;

    // no biases or velocities

    void init (layer_data data, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime);

    void feedforward(int previous_feature_maps, vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &z);

    void backprop(int previous_feature_maps, vector<float> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z);

    void update(hyperparams params);

};

struct flatten_layer : public layer {
    // number of neurons
    network_data n_in;
    network_data n_out;

    // no biases or velocities

    void init (layer_data data, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime);

    void feedforward(int previous_feature_maps, vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &z);

    void backprop(int previous_feature_maps, vector<float> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z);

    void update(hyperparams params);

};

struct input_layer : public layer {
    network_data n_out;
    // no biases or velocities

    void init (layer_data data, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime);

    void feedforward(int previous_feature_maps, vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &z);

    void backprop(int previous_feature_maps, vector<float> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z);

    void update(hyperparams params);

};

struct Network {
    // number of layers
    int L;

    // layers
    vector<unique_ptr<layer>> layers;

    // activation function
    function<double(double)> activationFunct;
    function<double(double)> activationFunctPrime;

    // cost function
    function<double(double, double)> costFunctPrime;

    void init (vector<layer_data> & layers, const function<double(double)> activationFunct, const function<double(double)> activationFunctPrime, const function<double(double, double)> costFunctPrime);

    pair<vector<vector<vector<vector<double>>>>, vector<vector<vector<vector<double>>>>> feedforward(vector<vector<double>> &a);

    void SGD(vector<pair<vector<vector<double>>, vector<double>>> training_data, vector<pair<vector<vector<double>>, vector<double>>> test_data, hyperparams params);

    void update_mini_batch(vector<pair<vector<vector<double>>, vector<double>>> &mini_batch, hyperparams params);

    void backprop(vector<vector<double>> &in, vector<double> &out);

    void save(string filename);

    void load(string filename);
};