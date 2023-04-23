hyperparams get_params() {
    hyperparams params;

    params.mini_batch_size = 10;
    params.epochs = 30;

    params.learning_rate = 0.5;
    params.L2_regularization_term = 0.9;
    params.momentum_coefficient = 0.25;

    return params;
}