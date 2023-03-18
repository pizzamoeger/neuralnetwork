#include <bits/stdc++.h>
#include "Network.h"

using namespace std;

default_random_engine generator;
normal_distribution<double> distribution(0.0, 1.0 / sqrt(784));

vector<pair<vector<double>, vector<double>>> load_data(string filename) {
    // loads data from csv file of form label, pixel1, pixel2, pixel3, ..., pixel784
    ifstream file;
    string line;

    file.open(filename);
    vector<pair<vector<double>, vector<double>>> data;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> input;
        vector<double> output (10, 0);

        int label = -1;
        while (ss.good()) {
            string substr;
            getline(ss, substr, ' ');
            if (label == -1) {
                label = stoi(substr);
            } else {
                input.push_back(stod(substr));
            }
        }
        output[label] = 1;
        data.push_back({input, output});
    }

    cerr << data.size() << " data loaded from " + filename + "\n";
    file.close();
    return data;
}

void initStructure() {
    ofstream file;

    cout << "num of hidden layers:";
    int L; cin >> L;
    L += 2;

    file.open("structure.txt");
    file << L << "\n";
    file << "784 ";

    cout << "sizes of hidden layers:";
    for (int i = 1; i < L-1; i++) {
        int x; cin >> x;
        file << x << " ";
    }

    file << "10\n";
    file.close();
}

void initBiases(vector<int> & sizes) {
    ofstream file;
    file.open("biases.txt");
    for (int i = 1; i < sizes.size(); i++) {
        for (int j = 0; j < sizes[i]; j++) {
            file << (double)distribution(generator) << " ";
        }
        file << "\n";
    }
    file.close();
}

void initWeights(vector<int> & sizes) {
    ofstream file;
    file.open("weights.txt");
    for (int i = 1; i < sizes.size(); i++) {
        for (int j = 0; j < sizes[i]; j++) {
            for (int k = 0; k < sizes[i-1]; k++) {
                file << (double)distribution(generator) << " ";
            }
            file << "^";
        }
        file << "\n";
    }
    file.close();
}


int main() {
    srand(time(NULL));

    cout << "create new network? (1/0):";
    bool nw; cin >> nw;

    // init the structure
    if (nw) initStructure();

    // load the structure
    ifstream file;
    file.open("structure.txt");
    int L; file >> L;
    vector<int> sizes (L);
    for (int i = 0; i < L; i++) file >> sizes[i];
    file.close();
    cerr << "Structure loaded\n";

    // init the biases and weights
    if (nw) {
        initBiases(sizes);
        initWeights(sizes);
    }

    // load the biases form biases.txt
    file.open("biases.txt");
    vector<vector<double>> biases;

    biases.push_back({});
    for (int i = 1; i < L; i++) {
        biases.push_back({});
        for (int j = 0; j < sizes[i]; j++) {
            double x; file >> x;
            biases[i].push_back(x);
        }
    }
    cerr << biases.size() << " biases loaded\n";
    file.close();

    // load the weights from weights.txt
    file.open("weights.txt");
    vector<vector<vector<double>>> weights;

    weights.push_back({});
    for (int i = 1; i < sizes.size(); i++) {
        weights.push_back({});
        for (int j = 0; j < sizes[i]; j++) {
            weights[i].push_back({});
            for (int k = 0; k < sizes[i-1]; k++) {
                double x; file >> x;
                weights[i][j].push_back(x);
            }
        }
    }
    cerr << weights.size() << " weights loaded\n";
    file.close();

    auto net = Network(sizes, biases, weights);
    cerr << "Network created\n";

    // train network
    auto test_data = load_data("mnist_test_normalized.data");
    cout << "train network? (1/0):";
    cin >> nw;

    if (nw) {
        auto training_data = load_data("mnist_train_normalized.data");

        int epochs; cout << "epochs:"; cin >> epochs; // 30
        int mini_batch_size; cout << "mini_batch_size:"; cin >> mini_batch_size; // 128
        double eta; cout << "eta:"; cin >> eta; // 0.5

        net.SGD(training_data, epochs, mini_batch_size, eta, test_data);

        // store the biases and weights in biases.txt and weights.txt
        ofstream file2;
        file2.open("biases.txt");
        for (int i = 1; i < net.biases.size(); i++) {
            for (int j = 0; j < net.biases[i].size(); j++) {
                file2 << net.biases[i][j] << " ";
            }
            file2 << "\n";
        }
        file2.close();
        file2.open("weights.txt");
        for (int i = 1; i < net.weights.size(); i++) {
            for (int j = 0; j < net.weights[i].size(); j++) {
                for (int k = 0; k < net.weights[i][j].size(); k++) {
                    file2 << net.weights[i][j][k] << " ";
                }
                file2 << "\n";
            }
            file2 << "\n";
        }
        file2.close();
    }

    // test network
    int correct = 0;
    for (int k = 0; k < test_data.size(); k++) {
        vector<double> output = net.feedforward(test_data[k].first);
        int max = 0;
        for (int j = 0; j < output.size(); j++) {
            if (output[j] > output[max]) max = j;
        }
        if (test_data[k].second[max] == 1) correct++;
    }
    cout << "Accuracy: " << (double)correct / test_data.size() << "\n";

    return 0;
}