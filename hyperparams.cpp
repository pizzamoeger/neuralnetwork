hyperparams get_params() {
    hyperparams params;

    params.mini_batch_size = 10;
    params.epochs = 30;

    params.learning_rate = 0.5;
    params.L2_regularization_term = 0.5;
    params.momentum_coefficent = 0.25;

    return params;
}