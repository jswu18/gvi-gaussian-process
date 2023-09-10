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
from experiments.shared.resolvers.kernel import (
    _resolve_ard_kernel,
    _resolve_nngp_kernel,
)
from src.inducing_points_selection import (
    ConditionalVarianceInducingPointsSelector,
    RandomInducingPointsSelector,
)

config.update("jax_enable_x64", True)

data_path = "experiments/classification/mnist/data"


ard_kernel, ard_kernel_parameters = _resolve_ard_kernel(
    kernel_parameters_config={
        "scaling": 1.0,
        "lengthscales": jnp.ones(
            784,
        ),
    },
    data_dimension=784,
)

nngp_kernel, nngp_kernel_parameters = _resolve_nngp_kernel(
    kernel_kwargs_config={
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
            "layer_9": {
                "layer_schema": "relu",
                "layer_kwargs": {},
            },
            "layer_10": {
                "layer_schema": "dense",
                "layer_kwargs": {"features": 1},
            },
        },
        "input_shape": [28, 28, 1],
    },
    data_dimension=784,
)

mlp_nngp_kernel, mlp_nngp_kernel_parameters = _resolve_nngp_kernel(
    kernel_kwargs_config={
        "layers": {
            "layer_1": {
                "layer_schema": "dense",
                "layer_kwargs": {"features": 1},
            },
            "layer_2": {
                "layer_schema": "relu",
                "layer_kwargs": {},
            },
        },
    },
    data_dimension=784,
)

train_data_percentage = 0.99
test_data_percentage = 0.005
validation_data_percentage = 0.005
number_of_inducing_per_label = 10
number_of_points_per_label = 5000
cmap = "binary"
full_random_inducing_points = np.ones((10, number_of_inducing_per_label, 28, 28))
full_ard_inducing_points = np.ones((10, number_of_inducing_per_label, 28, 28))
full_mlp_nngp_inducing_points = np.ones((10, number_of_inducing_per_label, 28, 28))
full_nngp_inducing_points = np.ones((10, number_of_inducing_per_label, 28, 28))

for digit in range(10):
    key = 0
    labels_to_include = [digit]
    number_of_labels = len(labels_to_include)

    mnist_data = MNIST(data_path)
    mnist_data.gz = True
    train_images, train_labels = mnist_data.load_training()
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
    x = experiment_data_list[0].full.x

    random_selector = RandomInducingPointsSelector()
    (
        random_inducing_points,
        random_inducing_points_idx,
    ) = random_selector.compute_inducing_points(
        key=key,
        training_inputs=x,
        number_of_inducing_points=number_of_inducing_per_label,
    )
    random_inducing_points = random_inducing_points.reshape(-1, 28, 28)
    full_random_inducing_points[digit, ...] = random_inducing_points

    fig, ax = plt.subplots(
        nrows=1,
        ncols=number_of_inducing_per_label,
        figsize=(number_of_inducing_per_label, 1),
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(number_of_inducing_per_label):
        ax[i].imshow(random_inducing_points[i, :, :], cmap=cmap)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
    plt.suptitle("Random Inducing Point Selection", y=1.1)
    fig.savefig(f"random_mnist_inducing_point_selection_{digit}", bbox_inches="tight")
    plt.show()

    selector = ConditionalVarianceInducingPointsSelector()
    ard_inducing_points, inducing_points_idx = selector.compute_inducing_points(
        key=key,
        training_inputs=x,
        number_of_inducing_points=number_of_inducing_per_label,
        kernel=ard_kernel,
        kernel_parameters=ard_kernel_parameters,
    )
    ard_inducing_points = ard_inducing_points.reshape(-1, 28, 28)
    full_ard_inducing_points[digit, ...] = ard_inducing_points

    fig, ax = plt.subplots(
        nrows=1,
        ncols=number_of_inducing_per_label,
        figsize=(number_of_inducing_per_label, 1),
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(number_of_inducing_per_label):
        ax[i].imshow(ard_inducing_points[i, :, :], cmap=cmap)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
    plt.suptitle("Greedy Variance Inducing Point Selection with ARD Kernel", y=1.1)
    fig.savefig(
        f"greedy_mnist_ard_inducing_point_selection_{digit}", bbox_inches="tight"
    )
    plt.show()

    selector = ConditionalVarianceInducingPointsSelector()
    mlp_nngp_inducing_points, inducing_points_idx = selector.compute_inducing_points(
        key=key,
        training_inputs=x,
        number_of_inducing_points=number_of_inducing_per_label,
        kernel=mlp_nngp_kernel,
        kernel_parameters=mlp_nngp_kernel_parameters,
    )
    mlp_nngp_inducing_points = mlp_nngp_inducing_points.reshape(-1, 28, 28)
    full_mlp_nngp_inducing_points[digit, ...] = mlp_nngp_inducing_points

    fig, ax = plt.subplots(
        nrows=1,
        ncols=number_of_inducing_per_label,
        figsize=(number_of_inducing_per_label, 1),
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(number_of_inducing_per_label):
        ax[i].imshow(mlp_nngp_inducing_points[i, :, :], cmap=cmap)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
    plt.suptitle("Greedy Variance Inducing Point Selection with MLP NNGP Kernel", y=1.1)
    fig.savefig(
        f"greedy_mnist_mlp_nngp_inducing_point_selection_{digit}", bbox_inches="tight"
    )

    plt.show()

    selector = ConditionalVarianceInducingPointsSelector()
    nngp_inducing_points, inducing_points_idx = selector.compute_inducing_points(
        key=key,
        training_inputs=x,
        number_of_inducing_points=number_of_inducing_per_label,
        kernel=nngp_kernel,
        kernel_parameters=nngp_kernel_parameters,
    )
    nngp_inducing_points = nngp_inducing_points.reshape(-1, 28, 28)
    full_nngp_inducing_points[digit, ...] = nngp_inducing_points

    fig, ax = plt.subplots(
        nrows=1,
        ncols=number_of_inducing_per_label,
        figsize=(number_of_inducing_per_label, 1),
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(number_of_inducing_per_label):
        ax[i].imshow(nngp_inducing_points[i, :, :], cmap=cmap)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
    plt.suptitle("Greedy Variance Inducing Point Selection with CNN NNGP Kernel", y=1.1)
    fig.savefig(
        f"greedy_mnist_cnn_nngp_inducing_point_selection_{digit}", bbox_inches="tight"
    )
    plt.show()


y_shift = 0.93
fontsize = "35"
fig, ax = plt.subplots(
    nrows=10,
    ncols=number_of_inducing_per_label,
    figsize=(number_of_inducing_per_label, 10),
)
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(10):
    for j in range(number_of_inducing_per_label):
        ax[i][j].imshow(full_random_inducing_points[i, j, :, :], cmap=cmap)
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)
        ax[i][j].spines["top"].set_visible(False)
        ax[i][j].spines["right"].set_visible(False)
        ax[i][j].spines["bottom"].set_visible(False)
        ax[i][j].spines["left"].set_visible(False)
plt.suptitle("Random Selection", y=y_shift, fontsize=fontsize)
fig.savefig(f"random_mnist_inducing_point_selection", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(
    nrows=10,
    ncols=number_of_inducing_per_label,
    figsize=(number_of_inducing_per_label, 10),
)
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(10):
    for j in range(number_of_inducing_per_label):
        ax[i][j].imshow(full_ard_inducing_points[i, j, :, :], cmap=cmap)
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)
        ax[i][j].spines["top"].set_visible(False)
        ax[i][j].spines["right"].set_visible(False)
        ax[i][j].spines["bottom"].set_visible(False)
        ax[i][j].spines["left"].set_visible(False)
plt.suptitle("Greedy Variance: ARD", y=y_shift, fontsize=fontsize)
fig.savefig(f"greedy_mnist_ard_inducing_point_selection", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(
    nrows=10,
    ncols=number_of_inducing_per_label,
    figsize=(number_of_inducing_per_label, 10),
)
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(10):
    for j in range(number_of_inducing_per_label):
        ax[i][j].imshow(full_mlp_nngp_inducing_points[i, j, :, :], cmap=cmap)
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)
        ax[i][j].spines["top"].set_visible(False)
        ax[i][j].spines["right"].set_visible(False)
        ax[i][j].spines["bottom"].set_visible(False)
        ax[i][j].spines["left"].set_visible(False)
plt.suptitle("Greedy Variance: FCN-NNGP", y=y_shift, fontsize=fontsize)
fig.savefig(f"greedy_mnist_fcn_nngp_inducing_point_selection", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(
    nrows=10,
    ncols=number_of_inducing_per_label,
    figsize=(number_of_inducing_per_label, 10),
)
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(10):
    for j in range(number_of_inducing_per_label):
        ax[i][j].imshow(full_nngp_inducing_points[i, j, :, :], cmap=cmap)
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)
        ax[i][j].spines["top"].set_visible(False)
        ax[i][j].spines["right"].set_visible(False)
        ax[i][j].spines["bottom"].set_visible(False)
        ax[i][j].spines["left"].set_visible(False)
plt.suptitle("Greedy Variance: CNN-NNGP", y=y_shift, fontsize=fontsize)
fig.savefig(f"greedy_mnist_cnn_nngp_inducing_point_selection", bbox_inches="tight")
plt.show()
