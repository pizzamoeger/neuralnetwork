#include "includes.h"

// sigmoid function and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoidPrime(double x) {
    return (sigmoid(x)*(1-sigmoid(x)));
}

// cross entropy cost function
double crossEntropyPrime(double output_activation, double y) {
    return (output_activation-y);
}

// load data
vector<pair<vector<vector<double>>, vector<double>>> load_data(string filename) {
    // loads data from csv file of form label, pixel1, pixel2, pixel3, ..., pixel784
    ifstream file;
    string line;

    file.open(filename);
    vector<pair<vector<vector<double>>, vector<double>>> data;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<vector<double>> input;
        vector<double> output (10, 0);

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

    params.mini_batch_size = 32;
    params.epochs = 30;

    params.learning_rate = 1;
    params.L2_regularization_term = 0;
    params.momentum_coefficient = 0;

    return params;
}
