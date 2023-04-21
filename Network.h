using namespace std;

#define double float

struct hyperparams {
    int learning_rate;
    int L2_regularization_term;
    int momentum_coefficent;
    int epochs;
    int mini_batch_size;
    int training_data_size;
    int test_data_size;
};

struct fullyConnectedLayer {
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

    vector<double> feedforward(vector<double> & a);

    void update(hyperparams params);

    void backprop(vector<double> & delta, vector<double> & activations, vector<double> & z);

};

struct Network {
    // number of layers
    int L;

    // sizes of the layers
    vector<int> sizes;

    // layers
    vector<fullyConnectedLayer> layers;

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