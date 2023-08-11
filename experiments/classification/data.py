from typing import List

import jax.numpy as jnp
import sklearn

from experiments.shared.data import ExperimentData, set_up_experiment


def set_up_classification_experiment_data(
    key,
    train_images,
    train_labels,
    one_hot_encoded_labels,
    number_of_points_per_label: int,
    labels_to_include,
    train_data_percentage: float,
    test_data_percentage: float,
    validation_data_percentage: float,
) -> List[ExperimentData]:
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
                train_data_percentage=train_data_percentage,
                test_data_percentage=test_data_percentage,
                validation_data_percentage=validation_data_percentage,
            )
        )
    return experiment_data_list


def one_hot_encode(y: jnp.ndarray, labels: List[int]) -> jnp.ndarray:
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(labels)
    return label_binarizer.transform(y)
