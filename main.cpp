#include "includes.h"
using namespace std;

int main() {
    srand(time(NULL));

    cout << "create new network? (1/0):";
    bool newNN; cin >> newNN;
    Network net;

    if (newNN) {
        int L; cout << "num of hidden layers: "; cin >> L;
        L += 2;
        vector<int> sizes (L); sizes[0] = 784; sizes[L-1] = 10;

        cout << "sizes of hidden layers: ";
        for (int i = 1; i < L-1; i++) cin >> sizes[i];

        net.init(sizes, sigmoid, sigmoidPrime, crossEntropyPrime);
    } else {
        string filename; cout << "filename: "; cin >> filename;
        vector<int> tmp = {};
        net.init(tmp, sigmoid, sigmoidPrime, crossEntropyPrime);
        net.load(filename);
    }

    // train network
    auto test_data = load_data("mnist_test_normalized.data");
    cout << "train network? (1/0):";
    bool train; cin >> train;

    if (train) {
        auto training_data = load_data("mnist_train_normalized.data");

        auto params = get_params();
        params.test_data_size  = test_data.size();
        params.training_data_size = training_data.size();

        net.SGD(training_data, test_data, params);

        bool save; cout << "save network? (1/0):"; cin >> save;
        if (save) {
            string filename; cout << "filename: "; cin >> filename;
            net.save(filename);
        }

        int correct = 0;
        for (int k = 0; k < training_data.size(); k++) {
            vector<double> output = net.feedforward(training_data[k].first).first[net.L-1];
            int max = 0;
            for (int j = 0; j < output.size(); j++) {
                if (output[j] > output[max]) max = j;
            }
            if (training_data[k].second[max] == 1) correct++;
        }
        cout << "accuracy in training data: " << (double)correct / training_data.size() << "\n";
    }

    // test network
    int correct = 0;
    for (int k = 0; k < test_data.size(); k++) {
        vector<double> output = net.feedforward(test_data[k].first).first[net.L-1];
        int max = 0;
        for (int j = 0; j < output.size(); j++) {
            if (output[j] > output[max]) max = j;
        }
        if (test_data[k].second[max] == 1) correct++;
    }
    cout << "general accuracy: " << (double)correct / test_data.size() << "\n";
}