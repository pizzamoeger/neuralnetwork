#include <bits/stdc++.h>
#include "Network.h"

using namespace std;

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
            getline(ss, substr, ',');
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

int main() {
    srand(time(NULL));
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 1.0 / sqrt(784));
    cout << "Do you want to create a new network? (1/0):";
    bool nw; cin >> nw;
    vector<int> sizes = {784, 30, 10};
    if (nw) {
        // print random biases in biases.txt and weights in weights.txt for the network
        ofstream file2;
        file2.open("biases.txt");
        for (int i = 1; i < sizes.size(); i++) {
            for (int j = 0; j < sizes[i]; j++) {
                file2 << (double)distribution(generator) << " ";
            }
            file2 << "\n";
        }
        file2.close();
        file2.open("weights.txt");
        for (int i = 1; i < sizes.size(); i++) {
            for (int j = 0; j < sizes[i]; j++) {
                for (int k = 0; k < sizes[i-1]; k++) {
                    file2 << (double)distribution(generator) << " ";
                }
                file2 << "^";
            }
            file2 << "\n";
        }
        file2.close();
    }

    // read the biases form biases.txt
    ifstream file;
    string line;
    file.open("biases.txt");
    vector<vector<double>> biases;
    biases.push_back({});
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> temp;
        //cerr << line<<"\n";
        while (ss.good()) {
            string substr;
            getline(ss, substr, ' ');
            if (substr == "") continue;
            temp.push_back(stod(substr));
        }
        biases.push_back(temp);
    }
    cerr << biases.size() << " biases loaded\n";
    file.close();

    // read the weights from weights.txt
    // it is of form weight[0][0][0], weight[0][0][1], weight[0][0][2], ..., \n weight[0][1][0], weight[0][1][1], weight[0][1][2], ..., \n \n weight[1][0][0], weight[1][0][1], weight[1][0][2], ..., \n weight[1][1][0], weight[1][1][1], weight[1][1][2], ..., \n \n ...

    file.open("weights.txt");
    vector<vector<vector<double>>> weights;
    weights.push_back({});
    while (getline(file, line)) {
        stringstream ss(line);
        vector<vector<double>> temp;
        while (ss.good()) {
            string substr;
            getline(ss, substr, '^');
            if (substr == "") continue;
            stringstream ss2(substr);
            vector<double> temp2;
            while (ss2.good()) {
                string substr2;
                getline(ss2, substr2, ' ');
                if (substr2 == "") continue;
                temp2.push_back(stod(substr2));
            }
            temp.push_back(temp2);
        }
        weights.push_back(temp);
    }
    cerr << weights.size() << " weights loaded\n";
    file.close();

    auto net = Network(sizes, biases, weights);
    cerr << "Network created\n";

    // train network
    auto test_data = load_data("mnist_test.csv");
    cout << "Do you want to train the network? (1/0):";
    cin >> nw;
    if (nw) {
        auto training_data = load_data("mnist_train.csv");
        net.SGD(training_data, 30, 128, 0.0001, test_data);

        // print the biases and weights in biases.txt and weights.txt
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
                file2 << "^";
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