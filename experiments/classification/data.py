import operator
from functools import reduce

import jax.numpy as jnp
import sklearn

from experiments.data import set_up_experiment
from src.kernels.base import KernelBase, KernelBaseParameters


def set_up_classification_experiment_data(
    key,
    train_images,
    train_labels,
    one_hot_encoded_labels,
    number_of_points_per_label: int,
    number_of_inducing_per_label: int,
    labels_to_include,
    train_data_percentage: float,
    test_data_percentage: float,
    validation_data_percentage: float,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
):
    label_dict = {}
    for label in labels_to_include:
        label_dict[label] = [
            idx for idx in range(len(train_labels)) if train_labels[idx] == label
        ]

    experiment_data_list = []
    for label in labels_to_include:
        experiment_data_list.append(
            set_up_experiment(
                key=key,
                x=jnp.array(
                    [
                        train_images[idx]
                        for idx in label_dict[label][:number_of_points_per_label]
                    ]
                ),
                y=jnp.array(
                    [
                        one_hot_encoded_labels[idx]
                        for idx in label_dict[label][:number_of_points_per_label]
                    ]
                ),
                number_of_inducing_points=number_of_inducing_per_label,
                train_data_percentage=train_data_percentage,
                test_data_percentage=test_data_percentage,
                validation_data_percentage=validation_data_percentage,
                kernel=kernel,
                kernel_parameters=kernel_parameters,
            )
        )
    experiment_data = reduce(operator.add, experiment_data_list)
    return experiment_data


def one_hot_encode(y, labels):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(labels)
    return label_binarizer.transform(y)
