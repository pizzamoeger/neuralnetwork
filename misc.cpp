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
    return max((float)0, x);
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
        for (int i = 0; i < 28 * 28; i++) data[lineIndex].first[i] = 0;

        int label = -1;
        int i = 0;
        while (ss.good()) {
            string substr;
            getline(ss, substr, ' ');
            if (label == -1) {
                label = stoi(substr);
            } else {
                if (i == 28 * 28) break;
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
    params.epochs = 30;

    params.fully_connected_weights_learning_rate = 0.04;
    params.fully_connected_biases_learning_rate = 0.02;
    params.convolutional_weights_learning_rate = 0.09;
    params.convolutional_biases_learning_rate = 0.02;

    params.L2_regularization_term = 0.0;
    params.momentum_coefficient = 0.0;

    return params;
}

void clear_data(data_point *data) {
    delete[] data;
}