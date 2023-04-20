#include "includes.h"

default_random_engine generator;

void fullyConnectedLayer::init (int n_in, int n_out, const function<double(double)>& activationFunct, const function<double(double)>& activationFunctPrime, const function<double(double, double)>& costFunctPrime) {
    normal_distribution<double> distribution(0.0, 1.0 / sqrt(n_in));

    this->n_in = n_in;
    this->n_out = n_out;
    this->activationFunct = activationFunct;
    this->activationFunctPrime = activationFunctPrime;
    this->costFunctPrime = costFunctPrime;

    biases.assign(n_out, distribution(generator));
    biasesVelocity.assign(n_out, 0);

    weights.resize(n_out);
    weightsVelocity.resize(n_out);
    for (int i = 0; i < n_out; i++) {
        weights[i].assign(n_in, distribution(generator));
        weightsVelocity[i].assign(n_in, 0);
    }

    updateB = vector<double> (n_out, 0);
    updateW = vector<vector<double>> (n_out, vector<double> (n_in, 0));
}

vector<double> fullyConnectedLayer::feedforward(vector<double> & a) {
    vector<double> z (n_out, 0);
    for (int i = 0; i < n_out; i++) {
        // get new activations
        for (int k = 0; k < n_in; k++) z[i] += weights[i][k]*a[k];
        z[i] += biases[i];
    }
    return z;
}

void fullyConnectedLayer::update(double learning_rate, double lambda, int n, double momentum_coefficient, int mini_batch_size) {
    // update velocities
    for (int j = 0; j < n_out; j++) {
        biasesVelocity[j] = momentum_coefficient*biasesVelocity[j]-(learning_rate/mini_batch_size)*updateB[j];
    }

    for (int j = 0; j < n_out; j++) {
        for (int k = 0; k < n_in; k++) {
            weightsVelocity[j][k] = momentum_coefficient*weightsVelocity[j][k]-(learning_rate/mini_batch_size)*updateW[j][k];
        }
    }

    // update weights and biases
    for (int j = 0; j < n_out; j++) {
        biases[j] = biases[j]+biasesVelocity[j];
    }
    for (int j = 0; j < n_out; j++) {
        for (int k = 0; k < n_in; k++) {
            weights[j][k] = (1-learning_rate*lambda/n)*weights[j][k]+weightsVelocity[j][k];
        }
    }

    updateB = vector<double> (n_out, 0);
    updateW = vector<vector<double>> (n_out, vector<double> (n_in, 0));
}

void fullyConnectedLayer::backprop(vector<double> & delta, vector<double> & activations, vector<double> & z) {

    for (int i = 0; i < n_out; i++) updateB[i] += delta[i];

    for (int i = 0; i < n_out; i++) {
        for (int j = 0; j < n_in; j++) {
            updateW[i][j] += delta[i]*activations[j];
        }
    }

    vector<double> newDelta (n_in, 0);

    for (int i = 0; i < n_out; i++) {
        for (int j = 0; j < n_in; j++) {
            newDelta[j] += delta[i]*weights[i][j];
            if (i == n_out-1) newDelta[j] *= activationFunctPrime(z[j]);
        }
    }
    delta = newDelta;
}