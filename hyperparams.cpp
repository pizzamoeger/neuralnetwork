hyperparams get_params() {
    hyperparams params;

    params.mini_batch_size = 10;
    params.epochs = 10;

    params.learning_rate = 1;
    params.L2_regularization_term = 0.5;
    params.momentum_coefficent = 0.5;

    params.training_data_size = 60000;
    params.test_data_size = 10000;

    return params;
}