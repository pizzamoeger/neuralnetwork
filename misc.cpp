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
                input.push_back(atof(substr.c_str()));
            }
        }
        output[label] = 1;
        data.push_back({input, output});
    }

    cerr << data.size() << " data loaded from " + filename + "\n";
    file.close();
    return data;
}
