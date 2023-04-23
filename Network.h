using namespace std;

#define double float

struct hyperparams {
    double learning_rate;
    double L2_regularization_term;
    double momentum_coefficient;
    int epochs;
    int mini_batch_size;
    int training_data_size;
    int test_data_size;
};

struct network_data {
    int x;
    int y;
};

struct fully_connected_layer {
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

    void init (int n_in, int n_out, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime);

    pair<vector<double>,vector<double>> feedforward(vector<double> & a);

    void backprop(vector<double> & delta, vector<double> & activations, vector<double> & z);

    void update(hyperparams params);

};

struct convolutional_layer {
    // number of neurons
    network_data n_in;
    network_data n_out;

    // cnn specific
    int stride_length;
    int receptive_field_length;
    int feature_maps;

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

    void init (network_data n_in, int stride_length, int receptive_field_length, int feature_maps, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime);

    pair<vector<vector<vector<double>>>,vector<vector<vector<double>>>> feedforward(int previous_feature_maps, vector<vector<vector<double>>> &a);

    void backprop(int previous_feature_maps, vector<float> &delta, vector<vector<vector<float>>> &activations, vector<vector<vector<float>>> &z);

    void update(hyperparams params);

};

struct Network {
    // number of layers
    int L;

    // sizes of the layers
    vector<int> sizes;

    // layers
    vector<fully_connected_layer> layers;

    // activation function
    function<double(double)> activationFunct;
    function<double(double)> activationFunctPrime;

    // cost function
    function<double(double, double)> costFunctPrime;

    void init (vector<int> & sizes, const function<double(double)> activationFunct, const function<double(double)> activationFunctPrime, const function<double(double, double)> costFunctPrime);

    pair<vector<vector<double>>, vector<vector<double>>> feedforward(vector<double> & a);

    void SGD(vector<pair<vector<double>,vector<double>>> training_data, vector<pair<vector<double>, vector<double>>> test_data, hyperparams params);

    void update_mini_batch(vector<pair<vector<double>,vector<double>>> & mini_batch, hyperparams params);

    void backprop(vector<double> & in, vector<double> & out);

    void save(string filename);

    void load(string filename);
};