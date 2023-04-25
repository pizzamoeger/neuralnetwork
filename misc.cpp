#include "includes.h"

// sigmoid function and its derivative
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}
float sigmoidPrime(float x) {
    return (sigmoid(x)*(1-sigmoid(x)));
}

// cross entropy cost function
float crossEntropyPrime(float output_activation, float y) {
    return (output_activation-y);
}

// load data
vector<pair<vector<vector<float>>, vector<float>>> load_data(string filename) {
    // loads data from csv file of form label, pixel1, pixel2, pixel3, ..., pixel784
    ifstream file;
    string line;

    file.open(filename);
    vector<pair<vector<vector<float>>, vector<float>>> data;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<vector<float>> input;
        vector<float> output (10, 0);

        int label = -1;
        int i = 0;
        int j = -1;
        while (ss.good()) {
            string substr;
            getline(ss, substr, ' ');
            if (label == -1) {
                label = stoi(substr);
            } else {
                if (i%28 == 0) {
                    i = 0;
                    j++;
                    input.push_back({});
                }
                i++;
                input[j].push_back(atof(substr.c_str()));
            }
        }
        output[label] = 1;
        data.push_back({input, output});
    }

    cerr << data.size() << " data loaded from " + filename + "\n";
    file.close();
    return data;
}

hyperparams get_params() {
    hyperparams params;

    params.mini_batch_size = 64;
    params.epochs = 50;

    params.fully_connected_weights_learning_rate = 0.1;
    params.fully_connected_biases_learning_rate = 0.8;
    params.convolutional_weights_learning_rate = 0.8;
    params.convolutional_biases_learning_rate = 0.8;

    params.L2_regularization_term = 0;
    params.momentum_coefficient = 0;

    return params;
}
