using namespace std;

#define double float

struct Network {
    // number of layers
    int L;

    // sizes of the layers
    vector<int> sizes;

    // biases[i][j] is bias of jth neuron in ith layer.
    // bias[0] = {}.
    vector<vector<double>> biases;
    vector<vector<double>> biasesVelocity;

    // weights[i][j][k] is weight of jth neuron in ith layer to kth neuron in i-1th layer.
    // weights[0] = {}
    vector<vector<vector<double>>> weights;
    vector<vector<vector<double>>> weightsVelocity;

    // activation function
    function<double(double)> activationFunct;
    function<double(double)> activationFunctPrime;

    // cost function
    function<double(double, double)> costFunctPrime;

    void init (vector<int> & sizes, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime);

    vector<double> feedforward(vector<double> & a);

    void SGD(vector<pair<vector<double>,vector<double>>> training_data, int epochs, int mini_batch_size, double learning_rate, vector<pair<vector<double>, vector<double>>> test_data, double lambda, double momentum_coefficient);

    void update_mini_batch(vector<pair<vector<double>,vector<double>>> & mini_batch, double learning_rate, double lambda, int n, double momentum_coefficient);

    pair<vector<vector<double>>, vector<vector<vector<double>>>> backprop(vector<double> & in, vector<double> & out);

    void save(string filename);

    void load(string filename);
};