if __name__ == "__main__":
    import os

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import orbax
    from experiments.nn_means import ConvNet, MultiLayerPerceptron
    from experiments.plotters import plot_losses, plot_two_losses
    from experiments.trainer import train_gvi, train_nll, train_tempered_nll
    from flax.training import orbax_utils
    from mnist import MNIST
    from neural_tangents import stax

    from experiments.classification.data import (
        one_hot_encode,
        set_up_classification_experiment_data,
    )
    from experiments.classification.plotters import plot_images
    from src.distributions import Multinomial
    from src.empirical_risks import NegativeLogLikelihood
    from src.generalised_variational_inference import GeneralisedVariationalInference
    from src.gps import ApproximateGPClassification, GPClassification
    from src.kernels import (
        CustomKernel,
        CustomKernelParameters,
        MultiOutputKernel,
        TemperedKernel,
        TemperedKernelParameters,
    )
    from src.kernels.approximate.svgp_diagonal_kernel import StochasticVariationalKernel
    from src.means import ConstantMean, ConstantMeanParameters, CustomMean
    from src.regularisations.point_wise import PointWiseWassersteinRegularisation

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    SEED = 0
    np.random.seed(SEED)
    KEY = jax.random.PRNGKey(SEED)

    number_of_points_per_label = 10
    labels_to_include = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    number_of_classes = len(labels_to_include)
    train_data_percentage = 0.8
    test_data_percentage = 0.1
    validation_data_percentage = 0.1
    number_of_inducing_per_label = int(
        jnp.sqrt(number_of_points_per_label * train_data_percentage)
    )

    # _, _, kernel_fn = stax.serial(
    #     stax.Dense(784, W_std=1, b_std=1),
    #     stax.Erf(),
    #     stax.Dense(1, W_std=1, b_std=1),
    # )

    _, _, kernel_fn = stax.serial(
        stax.Conv(32, (3, 3)),
        stax.Relu(),
        stax.AvgPool((2, 2), strides=(2, 2)),
        stax.Conv(64, (3, 3)),
        stax.Relu(),
        stax.AvgPool((2, 2), strides=(2, 2)),
        stax.Flatten(),
        stax.Dense(256),
        stax.Relu(),
        stax.Dense(1),
    )

    KERNEL = CustomKernel(
        kernel_function=lambda x1, x2: kernel_fn(x1, x2, "nngp"),
        # preprocess_function=lambda x: x.reshape(-1, 784),
        preprocess_function=lambda x: x.reshape(-1, 28, 28, 1),
    )
    KERNEL_PARAMETERS = CustomKernelParameters()

    MULTI_OUTPUT_KERNEL = MultiOutputKernel(
        kernels=[KERNEL for _ in range(number_of_classes)]
    )
    MULTI_OUTPUT_KERNEL_PARAMETERS = MULTI_OUTPUT_KERNEL.Parameters(
        kernels=[KERNEL_PARAMETERS] * number_of_classes,
    )

    # neural_network = MultiLayerPerceptron([1, 784, len(labels_to_include)])
    neural_network = ConvNet(number_of_outputs=len(labels_to_include))
    custom_mean = CustomMean(
        mean_function=lambda parameters, x: neural_network.apply(parameters, x),
        number_output_dimensions=len(labels_to_include),
        # preprocess_function=lambda x: x.reshape(-1, 784),
        preprocess_function=lambda x: x.reshape(-1, 28, 28, 1),
    )
    mnist_data = MNIST("mnist/data")
    mnist_data.gz = True
    TRAIN_IMAGES, TRAIN_LABELS = mnist_data.load_training()
    ONE_HOT_ENCODED_LABELS = one_hot_encode(
        y=jnp.array(TRAIN_LABELS).astype(int), labels=labels_to_include
    )
    output_folder = "mnist/outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    KEY, SUBKEY = jax.random.split(KEY)

    experiment_data = set_up_classification_experiment_data(
        key=SUBKEY,
        train_images=TRAIN_IMAGES,
        train_labels=TRAIN_LABELS,
        one_hot_encoded_labels=ONE_HOT_ENCODED_LABELS,
        number_of_points_per_label=number_of_points_per_label,
        number_of_inducing_per_label=number_of_inducing_per_label,
        labels_to_include=labels_to_include,
        train_data_percentage=train_data_percentage,
        test_data_percentage=test_data_percentage,
        validation_data_percentage=validation_data_percentage,
        kernel=KERNEL,
        kernel_parameters=KERNEL_PARAMETERS,
    )
    fig = plot_images(
        x=experiment_data.x_inducing,
        y=experiment_data.y_inducing,
        reshape_function=lambda x: x.reshape(28, 28),
    )
    fig.savefig(
        os.path.join(output_folder, f"inducing_points.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    gp = GPClassification(
        kernel=MULTI_OUTPUT_KERNEL,
        mean=ConstantMean(number_output_dimensions=len(labels_to_include)),
        x=experiment_data.x_inducing,
        y=experiment_data.y_inducing,
    )
    gp_parameters = gp.Parameters(
        log_observation_noise=jnp.log(jnp.ones(len(labels_to_include))),
        kernel=MULTI_OUTPUT_KERNEL_PARAMETERS,
        mean=ConstantMeanParameters(constant=jnp.zeros(len(labels_to_include))),
    )
    probabilities = Multinomial(
        **gp.predict_probability(
            parameters=gp_parameters, x=experiment_data.x_inducing
        ).dict()
    )
    print(probabilities.probabilities[0, :])
    KEY, SUBKEY = jax.random.split(KEY)
    training_epochs = 100
    lr = 1e-2
    save_checkpoint_frequency = 1000
    batch_size = 10
    nll_break_condition = 0
    gp_parameters, reference_losses = train_nll(
        key=SUBKEY,
        gp=gp,
        gp_parameters=gp_parameters,
        x=experiment_data.x_inducing,
        y=experiment_data.y_inducing,
        learning_rate=lr,
        number_of_epochs=training_epochs,
        save_checkpoint_frequency=save_checkpoint_frequency,
        batch_size=batch_size,
        checkpoint_path=os.path.join(
            output_folder, "training-checkpoints", "reference"
        ),
        nll_break_condition=nll_break_condition,
    )
    ckpt = gp_parameters.dict()
    save_args = orbax_utils.save_args_from_target(ckpt)
    reference_parameters_path = os.path.join(
        output_folder, "training-checkpoints", f"reference.ckpt"
    )

    orbax_checkpointer.save(
        reference_parameters_path, ckpt, save_args=save_args, force=True
    )
    fig = plot_losses(
        losses=reference_losses,
        loss_name="Negative Log Likelihood",
        title=f"Reference GP NLL Loss (MNIST)",
    )
    fig.savefig(
        os.path.join(output_folder, "reference-losses.png"), bbox_inches="tight"
    )
    plt.close(fig)
    np.save(
        os.path.join(output_folder, "training-checkpoints", "reference-losses.npy"),
        np.array(reference_losses),
    )
    probabilities = Multinomial(
        **gp.predict_probability(
            parameters=gp_parameters, x=experiment_data.x_inducing
        ).dict()
    )
    print(probabilities.probabilities[0, :])

    approximate_gp = ApproximateGPClassification(
        kernel=MultiOutputKernel(
            kernels=[
                StochasticVariationalKernel(
                    reference_kernel=reference_kernel,
                    reference_kernel_parameters=reference_kernel_parameters,
                    log_observation_noise=gp_parameters.log_observation_noise[i],
                    inducing_points=experiment_data.x_inducing,
                    training_points=experiment_data.x_train,
                    diagonal_regularisation=1e-8,
                )
                for i, (reference_kernel, reference_kernel_parameters) in enumerate(
                    zip(
                        MULTI_OUTPUT_KERNEL.kernels,
                        gp_parameters.kernel.kernels,
                    )
                )
            ]
        ),
        mean=custom_mean,
    )
    approximate_gp_parameters = approximate_gp.generate_parameters(
        {
            "mean": approximate_gp.mean.generate_parameters(
                {
                    "custom": neural_network.init(
                        SUBKEY,
                        approximate_gp.mean.preprocess_function(
                            experiment_data.x_train[:1, ...]
                        ),
                    )
                }
            ),
            "kernel": approximate_gp.kernel.initialise_random_parameters(SUBKEY),
        }
    )

    probabilities = Multinomial(
        **approximate_gp.predict_probability(
            parameters=approximate_gp_parameters, x=experiment_data.x_inducing
        ).dict()
    )
    print(probabilities.probabilities[0, :])

    KEY, SUBKEY = jax.random.split(KEY)
    regulariser = PointWiseWassersteinRegularisation
    regularisation = regulariser(
        gp=approximate_gp,
        regulariser=gp,
        regulariser_parameters=gp_parameters,
    )
    empirical_risk = NegativeLogLikelihood(
        gp=approximate_gp,
    )
    gvi = GeneralisedVariationalInference(
        regularisation=regularisation,
        empirical_risk=empirical_risk,
    )

    lr = 1e-2
    training_epochs = 50
    save_checkpoint_frequency = 1000
    batch_size = 10
    approximate_parameters_path = os.path.join(
        output_folder,
        "training-checkpoints",
        f"approximate-{regulariser.__name__}.ckpt",
    )
    approximate_gp_parameters, gvi_losses, emp_risk_losses, reg_losses = train_gvi(
        key=SUBKEY,
        gp_parameters=approximate_gp_parameters,
        gvi=gvi,
        x=experiment_data.x_train,
        y=experiment_data.y_train,
        learning_rate=lr,
        number_of_epochs=training_epochs,
        save_checkpoint_frequency=save_checkpoint_frequency,
        batch_size=batch_size,
        checkpoint_path=os.path.join(
            output_folder,
            "training-checkpoints",
            f"approximate-{regulariser.__name__}",
        ),
    )
    ckpt = approximate_gp_parameters.dict()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(
        approximate_parameters_path, ckpt, save_args=save_args, force=True
    )
    fig = plot_losses(
        losses=gvi_losses,
        loss_name="GVI Loss",
        title=f"GVI Loss (MNIST) ({regulariser.__name__})",
    )
    fig.savefig(
        os.path.join(
            output_folder, f"approximate-gvi-losses-{regulariser.__name__}.png"
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
    fig = plot_two_losses(
        loss1=emp_risk_losses,
        loss1_name="Empirical Risk",
        loss2=reg_losses,
        loss2_name="Regularisation",
        title=f"GVI Loss Decomposed (MNIST) ({regulariser.__name__})",
    )
    fig.savefig(
        os.path.join(
            output_folder,
            f"approximate-gvi-losses-breakdown-{regulariser.__name__}.png",
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
    probabilities = Multinomial(
        **approximate_gp.predict_probability(
            parameters=approximate_gp_parameters, x=experiment_data.x_inducing
        ).dict()
    )
    print(probabilities.probabilities[0, :])

    tempered_gp = type(approximate_gp)(
        mean=approximate_gp.mean,
        kernel=TemperedKernel(
            base_kernel=approximate_gp.kernel,
            base_kernel_parameters=approximate_gp_parameters.kernel,
            number_output_dimensions=approximate_gp.kernel.number_output_dimensions,
        ),
    )
    tempered_gp_parameters = tempered_gp.Parameters(
        log_observation_noise=approximate_gp_parameters.log_observation_noise,
        mean=approximate_gp_parameters.mean,
        kernel=TemperedKernelParameters(
            log_tempering_factor=jnp.log(
                jnp.ones(approximate_gp.kernel.number_output_dimensions)
            )
        ),
    )

    probabilities = Multinomial(
        **tempered_gp.predict_probability(
            parameters=tempered_gp_parameters, x=experiment_data.x_inducing
        ).dict()
    )
    print(probabilities.probabilities[0, :])

    parameters_path = os.path.join(
        output_folder, "training-checkpoints", f"tempered-{regulariser.__name__}.ckpt"
    )

    KEY, SUBKEY = jax.random.split(KEY)
    training_epochs = 10000
    lr = 1e-2
    save_checkpoint_frequency = 1000
    batch_size = 10
    tempered_gp_parameters, losses = train_tempered_nll(
        key=SUBKEY,
        gp=tempered_gp,
        gp_parameters=tempered_gp_parameters,
        base_gp_parameters=approximate_gp_parameters,
        x=experiment_data.x_validation,
        y=experiment_data.y_validation,
        learning_rate=lr,
        number_of_epochs=training_epochs,
        save_checkpoint_frequency=save_checkpoint_frequency,
        batch_size=batch_size,
        checkpoint_path=os.path.join(
            output_folder,
            "training-checkpoints",
            f"tempered-{regulariser.__name__}",
        ),
    )
    ckpt = tempered_gp_parameters.dict()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(parameters_path, ckpt, save_args=save_args, force=True)
    fig = plot_losses(
        losses=losses,
        loss_name="Negative Log Likelihood",
        title=f"Tempered Approximate GP NLL Loss (MNIST) ({regulariser.__name__})",
    )
    fig.savefig(
        os.path.join(output_folder, f"tempered-losses-{regulariser.__name__}.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    probabilities = Multinomial(
        **tempered_gp.predict_probability(
            parameters=tempered_gp_parameters, x=experiment_data.x_inducing
        ).dict()
    )
    print(probabilities.probabilities[0, :])
