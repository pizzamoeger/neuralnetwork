#include "includes.h"

// sigmoid function and its derivative
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}
float sigmoidPrime(float x) {
    return (sigmoid(x)*(1-sigmoid(x)));
}

const float al = 0;

float relu(float x){
    return max(x, 0.0f);
}

float reluPrime(float x){
    if (x > 0) return 1.0f;
    return al;
}

// cross entropy cost function
float crossEntropyPrime(float output_activation, float y) {
    return (output_activation-y);
}

// load data
pair<data_point*, int> load_data(string filename) {
    // loads data from csv file of form label, pixel1, pixel2, pixel3, ..., pixel784
    ifstream file;
    string line;

    file.open(filename);

    // how many lines there are in the file
    int dataPoints = 0;
    while (getline(file, line)) {
        dataPoints++;
    }
    file.close();

    file.open(filename);

    data_point *data = new data_point[dataPoints];
    int lineIndex = 0;

    while (getline(file, line)) {
        stringstream ss(line);

        for (int i = 0; i < 10; i++) data[lineIndex].second[i] = 0;

        int label = -1;
        int i = 0;
        while (ss.good()) {
            string substr;
            getline(ss, substr, ' ');
            if (label == -1) {
                label = stoi(substr);
            } else {
                data[lineIndex].first[i] = atof(substr.c_str());
                i++;
            }
        }
        data[lineIndex].second[label] = 1;
        lineIndex++;
    }
    cerr << dataPoints << " data loaded from " + filename + "\n";
    file.close();
    return {data, dataPoints};
}

hyperparams get_params() {
    hyperparams params;

    params.mini_batch_size = 16;
    params.epochs = 5;

    params.fully_connected_weights_learning_rate = 0.3520532880647171;
    params.fully_connected_biases_learning_rate = 0.4444444444444444;
    params.convolutional_weights_learning_rate = 8.268717492598576;
    params.convolutional_biases_learning_rate = 0.4444444444444444;

    params.L2_regularization_term = 0.0;
    params.momentum_coefficient = 0.0;

    return params;
}

void clear_data(data_point *data) {
    delete[] data;
}