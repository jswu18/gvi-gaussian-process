if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)
    NUMBER_OF_DATA_POINTS = 500
    SIGMA_TRUE = 0.5
    TRAIN_DATA_PERCENTAGE = 0.8
    NUMBER_OF_TEST_INTERVALS = 2
    TOTAL_NUMBER_OF_INTERVALS = 8
    NUMBER_OF_INDUCING_POINTS = int(np.sqrt(NUMBER_OF_DATA_POINTS))
    REFERENCE_GP_LR = 1e-3
    REFERENCE_GP_TRAINING_EPOCHS = 10000
    REFERENCE_SAVE_CHECKPOINT_FREQUENCY = 1000
    REFERENCE_GP_BATCH_SIZE = 100
    REFERENCE_LOAD_CHECKPOINT = False
    OUTPUT_DIRECTORY = "outputs"
    EL_MATRIX_LOWER_BOUND = 1e-5
    INCLUDE_EIGENDECOMPOSITION = False
    APPROXIMATE_GP_LR = 1e-5
    APPROXIMATE_GP_TRAINING_EPOCHS = 1000000
    APPROXIMATE_SAVE_CHECKPOINT_FREQUENCY = 1000
    APPROXIMATE_GP_BATCH_SIZE = 500
    APPROXIMATE_LOAD_CHECKPOINT = False
    TEMPERED_GP_LR = 1e-3
    TEMPERED_GP_TRAINING_EPOCHS = 2000
    TEMPERED_SAVE_CHECKPOINT_FREQUENCY = 1000
    TEMPERED_GP_BATCH_SIZE = 500
    TEMPERED_LOAD_CHECKPOINT = False
    X = jnp.linspace(-2, 2, NUMBER_OF_DATA_POINTS, dtype=np.float64).reshape(-1, 1)

    _, _, kernel_fn = stax.serial(
        stax.Dense(10, W_std=10, b_std=10),
        stax.Erf(),
        stax.Dense(1, W_std=10, b_std=10),
    )
    KERNEL = CustomKernel(lambda x1, x2: kernel_fn(x1, x2, "nngp"))
    KERNEL_PARAMETERS = KERNEL.Parameters()
    NEURAL_NETWORK = MultiLayerPerceptron([1, 10, 1])

    for CURVE_FUNCTION in CURVE_FUNCTIONS:
        np.random.seed(CURVE_FUNCTION.seed)
        KEY, SUBKEY = jax.random.split(jax.random.PRNGKey(CURVE_FUNCTION.seed))
        run_experiment(
            key=SUBKEY,
            curve_function=CURVE_FUNCTION,
            x=X,
            sigma_true=SIGMA_TRUE,
            number_of_test_intervals=NUMBER_OF_TEST_INTERVALS,
            total_number_of_intervals=TOTAL_NUMBER_OF_INTERVALS,
            number_of_inducing_points=NUMBER_OF_INDUCING_POINTS,
            train_data_percentage=TRAIN_DATA_PERCENTAGE,
            kernel=KERNEL,
            kernel_parameters=KERNEL_PARAMETERS,
            reference_gp_lr=REFERENCE_GP_LR,
            reference_gp_training_epochs=REFERENCE_GP_TRAINING_EPOCHS,
            reference_save_checkpoint_frequency=REFERENCE_SAVE_CHECKPOINT_FREQUENCY,
            reference_gp_batch_size=REFERENCE_GP_BATCH_SIZE,
            reference_load_checkpoint=REFERENCE_LOAD_CHECKPOINT,
            approximate_gp_lr=APPROXIMATE_GP_LR,
            approximate_gp_training_epochs=APPROXIMATE_GP_TRAINING_EPOCHS,
            approximate_save_checkpoint_frequency=APPROXIMATE_SAVE_CHECKPOINT_FREQUENCY,
            approximate_gp_batch_size=APPROXIMATE_GP_BATCH_SIZE,
            approximate_load_checkpoint=APPROXIMATE_LOAD_CHECKPOINT,
            output_directory=OUTPUT_DIRECTORY,
            neural_network=NEURAL_NETWORK,
            el_matrix_lower_bound=EL_MATRIX_LOWER_BOUND,
            include_eigendecomposition=INCLUDE_EIGENDECOMPOSITION,
            tempered_gp_lr=TEMPERED_GP_LR,
            tempered_gp_training_epochs=TEMPERED_GP_TRAINING_EPOCHS,
            tempered_save_checkpoint_frequency=TEMPERED_SAVE_CHECKPOINT_FREQUENCY,
            tempered_gp_batch_size=TEMPERED_GP_BATCH_SIZE,
            tempered_load_checkpoint=TEMPERED_LOAD_CHECKPOINT,
        )
