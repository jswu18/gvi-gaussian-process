import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.config import config
from mnist import MNIST

from experiments.classification.data import (
    one_hot_encode,
    set_up_classification_experiment_data,
)
from experiments.shared.resolvers.kernel import (
    _resolve_nngp_kernel,
    _resolve_ard_kernel,
)
from src.inducing_points_selection import (
    ConditionalVarianceInducingPointsSelector,
    RandomInducingPointsSelector,
)

config.update("jax_enable_x64", True)
image_output_path = ""
data_path = "data"
labels_to_include = [7]
train_data_percentage = 0.8
test_data_percentage = 0.1
validation_data_percentage = 0.1
number_of_inducing_per_label = 10
number_of_points_per_label = 10000
key = 0


(
    ard_kernel,
    ard_kernel_parameters,
) = _resolve_ard_kernel(
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
fig, ax = plt.subplots(
    nrows=1,
    ncols=number_of_inducing_per_label,
    figsize=(number_of_inducing_per_label, 1),
)
for i in range(number_of_inducing_per_label):
    ax[i].imshow(random_inducing_points[i, :, :])
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)
fig.tight_layout()
plt.suptitle("Random Inducing Point Selection", y=1.1)
fig.savefig(
    os.path.join(image_output_path, "random_mnist_inducing_point_selection"),
    bbox_inches="tight",
)

selector = ConditionalVarianceInducingPointsSelector()
mlp_nngp_inducing_points, inducing_points_idx = selector.compute_inducing_points(
    key=key,
    training_inputs=x,
    number_of_inducing_points=number_of_inducing_per_label,
    kernel=mlp_nngp_kernel,
    kernel_parameters=mlp_nngp_kernel_parameters,
)
mlp_nngp_inducing_points = mlp_nngp_inducing_points.reshape(-1, 28, 28)

fig, ax = plt.subplots(
    nrows=1,
    ncols=number_of_inducing_per_label,
    figsize=(number_of_inducing_per_label, 1),
)
for i in range(number_of_inducing_per_label):
    ax[i].imshow(mlp_nngp_inducing_points[i, :, :])
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)
fig.tight_layout()
plt.suptitle("Greedy Variance Inducing Point Selection with MLP NNGP Kernel", y=1.1)
fig.savefig(
    os.path.join(image_output_path, "greedy_mnist_mlp_nngp_inducing_point_selection"),
    bbox_inches="tight",
)

selector = ConditionalVarianceInducingPointsSelector()
nngp_inducing_points, inducing_points_idx = selector.compute_inducing_points(
    key=key,
    training_inputs=x,
    number_of_inducing_points=number_of_inducing_per_label,
    kernel=nngp_kernel,
    kernel_parameters=nngp_kernel_parameters,
)
nngp_inducing_points = nngp_inducing_points.reshape(-1, 28, 28)
fig, ax = plt.subplots(
    nrows=1,
    ncols=number_of_inducing_per_label,
    figsize=(number_of_inducing_per_label, 1),
)
for i in range(number_of_inducing_per_label):
    ax[i].imshow(nngp_inducing_points[i, :, :])
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)
fig.tight_layout()
plt.suptitle("Greedy Variance Inducing Point Selection with CNN NNGP Kernel", y=1.1)
fig.savefig(
    os.path.join(image_output_path, "greedy_mnist_cnn_nngp_inducing_point_selection"),
    bbox_inches="tight",
)


selector = ConditionalVarianceInducingPointsSelector()
ard_inducing_points, inducing_points_idx = selector.compute_inducing_points(
    key=key,
    training_inputs=x,
    number_of_inducing_points=number_of_inducing_per_label,
    kernel=ard_kernel,
    kernel_parameters=ard_kernel_parameters,
)
ard_inducing_points = ard_inducing_points.reshape(-1, 28, 28)
fig, ax = plt.subplots(
    nrows=1,
    ncols=number_of_inducing_per_label,
    figsize=(number_of_inducing_per_label, 1),
)
for i in range(number_of_inducing_per_label):
    ax[i].imshow(ard_inducing_points[i, :, :])
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)
fig.tight_layout()
plt.suptitle("Greedy Variance Inducing Point Selection with ARD Kernel", y=1.1)
fig.savefig(
    os.path.join(image_output_path, "greedy_mnist_ard_inducing_point_selection"),
    bbox_inches="tight",
)
