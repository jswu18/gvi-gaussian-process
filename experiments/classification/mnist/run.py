

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    import numpy as np
    from mnist import MNIST
    from neural_tangents import stax
    from experiments.neural_networks import MultiLayerPerceptron
    from src.kernels import CustomKernel
    from experiments.classification.data import set_up_classification_experiment_data

    SEED = 0
    np.random.seed(SEED)
    KEY = jax.random.PRNGKey(SEED)

    number_of_points_per_label = 50
    number_of_inducing_per_label = int(jnp.sqrt(number_of_points_per_label * 0.7))
    labels_to_include = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    number_of_classes = len(labels_to_include)
    train_data_percentage = 0.8
    test_data_percentage = 0.1
    validation_data_percentage = 0.1

    _, _, kernel_fn = stax.serial(
        stax.Dense(784, W_std=15, b_std=15),
        stax.Erf(),
        stax.Dense(1, W_std=15, b_std=15),
    )

    # _, _, kernel_fn = stax.serial(
    #     stax.Conv(32, (3, 3)),
    #     stax.Relu(),
    #     stax.AvgPool((2, 2), strides=(2, 2)),
    #     stax.Conv(64, (3, 3)),
    #     stax.Relu(),
    #     stax.AvgPool((2, 2), strides=(2, 2)),
    #     stax.Flatten(),
    #     stax.Dense(256),
    #     stax.Relu(),
    #     stax.Dense(1),
    # )

    KERNEL = CustomKernel(
        kernel_function=lambda x1, x2: kernel_fn(x1, x2, "nngp"),
        preprocess_function=lambda x: x.reshape(-1, 784),
    )
    KERNEL_PARAMETERS = KERNEL.Parameters()
    NEURAL_NETWORK = MultiLayerPerceptron([1, 784, 10])

    mnist_data = MNIST("mnist_data")
    mnist_data.gz = True
    TRAIN_IMAGES, TRAIN_LABELS = mnist_data.load_training()
    KEY, SUBKEY = jax.random.split(KEY)

    experiment_data = set_up_classification_experiment_data(
        key=SUBKEY,
        train_images=TRAIN_IMAGES,
        train_labels=TRAIN_LABELS,
        number_of_points_per_label=number_of_points_per_label,
        number_of_inducing_per_label=number_of_inducing_per_label,
        labels_to_include=labels_to_include,
        train_data_percentage=train_data_percentage,
        test_data_percentage=test_data_percentage,
        validation_data_percentage=validation_data_percentage,
        kernel=KERNEL,
        kernel_parameters=KERNEL_PARAMETERS,
    )
    assert True