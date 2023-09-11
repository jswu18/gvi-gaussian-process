import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.config import config
from mnist import MNIST

from experiments.classification.data import (
    one_hot_encode,
    set_up_classification_experiment_data,
)
from experiments.shared.data import Data
from experiments.shared.resolvers import empirical_risk_resolver
from experiments.shared.resolvers.kernel import _resolve_custom_mapping_kernel
from experiments.shared.trainer import Trainer, TrainerSettings
from src.gps import GPClassification
from src.inducing_points_selection import ConditionalVarianceInducingPointsSelector
from src.kernels import MultiOutputKernel, MultiOutputKernelParameters
from src.means import ConstantMean, ConstantMeanParameters

config.update("jax_enable_x64", True)


data_path = "experiments/classification/mnist/data"

kernel_kwargs_config = {
    "diagonal_regularisation": 1.0e-5,
    "is_diagonal_regularisation_absolute_scale": False,
    "base_kernel": {
        "kernel_schema": "polynomial",
        "kernel_kwargs": {"polynomial_degree": 1},
        "kernel_parameters": {
            "constant": 10000.0,
            "scaling": 10000.0,
        },
    },
    "nn_function_kwargs": {
        "seed": 0,
        "input_shape": [28, 28, 1],
        "layers": {
            "layer_1": {
                "layer_schema": "convolution",
                "layer_kwargs": {
                    "features": 32,
                    "kernel_size": [3, 3],
                },
            },
            "layer_2": {
                "layer_schema": "relu",
                "layer_kwargs": {},
            },
            "layer_3": {
                "layer_schema": "average_pool",
                "layer_kwargs": {"window_shape": [2, 2], "strides": [2, 2]},
            },
            "layer_4": {
                "layer_schema": "convolution",
                "layer_kwargs": {
                    "features": 64,
                    "kernel_size": [3, 3],
                },
            },
            "layer_5": {
                "layer_schema": "relu",
                "layer_kwargs": {},
            },
            "layer_6": {
                "layer_schema": "average_pool",
                "layer_kwargs": {"window_shape": [2, 2], "strides": [2, 2]},
            },
            "layer_7": {
                "layer_schema": "flatten",
                "layer_kwargs": {},
            },
            "layer_8": {
                "layer_schema": "dense",
                "layer_kwargs": {"features": 256},
            },
        },
    },
}

nn_kernel_list = []
nn_kernel_parameters_list = []
for digit in range(10):
    nn_kernel, nn_kernel_parameters = _resolve_custom_mapping_kernel(
        kernel_kwargs_config=kernel_kwargs_config,
        data_dimension=784,
    )
    nn_kernel_list.append(nn_kernel)
    nn_kernel_parameters_list.append(nn_kernel_parameters)

classifier_kernel = MultiOutputKernel(kernels=nn_kernel_list)
classifier_kernel_parameters = MultiOutputKernelParameters(
    kernels=nn_kernel_parameters_list
)

classifier_mean = ConstantMean(number_output_dimensions=10)
classifier_mean_parameters = ConstantMeanParameters(constant=jnp.zeros((10,)))
train_data_percentage = 0.99
test_data_percentage = 0.005
validation_data_percentage = 0.005
number_of_inducing_per_label = 10
number_of_points_per_label = 5000
cmap = "binary"
full_nn_inducing_points = np.ones((10, number_of_inducing_per_label, 28, 28))
full_nn_inducing_points_responses = np.ones((10, number_of_inducing_per_label, 10))

key = 0

mnist_data = MNIST(data_path)
mnist_data.gz = True
train_images, train_labels = mnist_data.load_training()
labels_to_include = sorted(list(set(train_labels)))
number_of_labels = len(labels_to_include)
one_hot_encoded_labels = one_hot_encode(
    y=jnp.array(train_labels).astype(int), labels=labels_to_include
)
key, subkey = jax.random.split(jax.random.PRNGKey(key))
experiment_data_list = set_up_classification_experiment_data(
    key=subkey,
    train_images=train_images,
    train_labels=train_labels,
    one_hot_encoded_labels=one_hot_encoded_labels,
    number_of_points_per_label=number_of_points_per_label,
    labels_to_include=labels_to_include,
    train_data_percentage=train_data_percentage,
    test_data_percentage=test_data_percentage,
    validation_data_percentage=validation_data_percentage,
    name="",
)
trainer_settings = TrainerSettings(
    seed=0,
    optimiser_schema="adabelief",
    learning_rate=1e-2,
    number_of_epochs=300,
    batch_size=1000,
    batch_shuffle=True,
    batch_drop_last=False,
)
y_shift = 0.93
fontsize = "35"

post_epoch_histories = []
inducing_points_list = []

for iteration in range(3):
    for digit in range(10):
        x = experiment_data_list[digit].full.x.reshape(-1, 28, 28, 1)
        selector = ConditionalVarianceInducingPointsSelector()
        nn_inducing_points, inducing_points_idx = selector.compute_inducing_points(
            key=key,
            training_inputs=x,
            number_of_inducing_points=number_of_inducing_per_label,
            kernel=classifier_kernel.kernels[digit],
            kernel_parameters=classifier_kernel_parameters.kernels[digit],
        )
        nn_inducing_points = nn_inducing_points.reshape(-1, 28, 28)
        full_nn_inducing_points[digit, ...] = nn_inducing_points
        full_nn_inducing_points_responses[digit, ...] = experiment_data_list[
            digit
        ].full.y[inducing_points_idx, ...]
    inducing_points_list.append(full_nn_inducing_points)
    fig, ax = plt.subplots(
        nrows=10,
        ncols=number_of_inducing_per_label,
        figsize=(number_of_inducing_per_label, 10),
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(10):
        for j in range(number_of_inducing_per_label):
            ax[i][j].imshow(full_nn_inducing_points[i, j, :, :], cmap=cmap)
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
            ax[i][j].spines["top"].set_visible(False)
            ax[i][j].spines["right"].set_visible(False)
            ax[i][j].spines["bottom"].set_visible(False)
            ax[i][j].spines["left"].set_visible(False)
    plt.suptitle(f"CNN Mapping ({iteration=})", y=y_shift, fontsize=fontsize)
    fig.savefig(
        f"cnn_mapping_kernel_inducing_point_selection_{iteration}", bbox_inches="tight"
    )
    plt.show()
    classifier_gp = GPClassification(
        mean=classifier_mean,
        kernel=classifier_kernel,
        x=full_nn_inducing_points.reshape(-1, 28, 28, 1),
        y=full_nn_inducing_points_responses.reshape(-1, 10),
    )
    classifier_gp_parameters = classifier_gp.generate_parameters(
        {
            "log_observation_noise": jnp.log(jnp.ones((10,))),
            "mean": classifier_mean_parameters.dict(),
            "kernel": classifier_kernel_parameters.dict(),
        }
    )
    data = Data(x=classifier_gp.x, y=classifier_gp.y)
    empirical_risk = empirical_risk_resolver(
        empirical_risk_schema="negative_log_likelihood",
        gp=classifier_gp,
    )
    trainer = Trainer(
        save_checkpoint_frequency=False,
        checkpoint_path="",
        post_epoch_callback=lambda parameters: {
            "empirical-risk": empirical_risk.calculate_empirical_risk(
                parameters, data.x, data.y
            )
        },
        break_condition_function=(
            lambda parameters: empirical_risk.calculate_empirical_risk(
                parameters, data.x, data.y
            )
            < -10000
        ),
    )
    classifier_gp_parameters, post_epoch_history = trainer.train(
        trainer_settings=trainer_settings,
        parameters=classifier_gp_parameters,
        data=data,
        loss_function=lambda parameters_dict, x, y: empirical_risk.calculate_empirical_risk(
            parameters=parameters_dict,
            x=x,
            y=y,
        ),
    )
    post_epoch_histories.append(post_epoch_history)
    classifier_mean_parameters = classifier_mean.generate_parameters(
        classifier_gp_parameters.mean
    )
    classifier_kernel_parameters = classifier_kernel.generate_parameters(
        classifier_gp_parameters.kernel
    )
iteration += 1
fig, ax = plt.subplots(
    nrows=10,
    ncols=number_of_inducing_per_label,
    figsize=(number_of_inducing_per_label, 10),
)
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(10):
    for j in range(number_of_inducing_per_label):
        ax[i][j].imshow(full_nn_inducing_points[i, j, :, :], cmap=cmap)
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)
        ax[i][j].spines["top"].set_visible(False)
        ax[i][j].spines["right"].set_visible(False)
        ax[i][j].spines["bottom"].set_visible(False)
        ax[i][j].spines["left"].set_visible(False)
plt.suptitle(f"CNN Mapping ({iteration=})", y=y_shift, fontsize=fontsize)
fig.savefig(
    f"cnn_mapping_kernel_inducing_point_selection_{iteration}", bbox_inches="tight"
)
plt.show()
