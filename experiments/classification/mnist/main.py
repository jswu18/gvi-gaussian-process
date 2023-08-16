import operator
import os
from functools import reduce
from typing import List

import jax
import jax.numpy as jnp
from mnist import MNIST

from experiments.classification.data import (
    one_hot_encode,
    set_up_classification_experiment_data,
)
from experiments.classification.metrics import calculate_metric
from experiments.classification.plotters import plot_images
from experiments.classification.runners import meta_train_reference_gp
from experiments.classification.schemes import ClassificationMetricScheme
from experiments.shared.nn_means import CNN, MLP
from experiments.shared.nngp_kernels import CNNGPKernel, MLPGPKernel
from experiments.shared.plotters import plot_losses, plot_two_losses
from experiments.shared.schemas import (
    EmpiricalRiskScheme,
    OptimiserScheme,
    RegularisationScheme,
)
from experiments.shared.trainer import TrainerSettings
from experiments.shared.trainers import train_approximate_gp, train_tempered_gp
from src.gps import ApproximateGPClassification
from src.kernels import CustomKernel, MultiOutputKernel, TemperedKernel
from src.kernels.approximate.svgp.kernelised_svgp_kernel import KernelisedSVGPKernel
from src.means import CustomMean

jax.config.update("jax_enable_x64", True)


# Experiment settings
key = 0
output_directory = "outputs"
checkpoints_folder_name = "training-checkpoints"
number_of_points_per_label = 1000

data_path = "data"
labels_to_include = [0, 1, 2]
train_data_percentage = 0.8
test_data_percentage = 0.1
validation_data_percentage = 0.1
number_of_inducing_per_label = int(
    jnp.sqrt(number_of_points_per_label * train_data_percentage)
)


reference_number_of_iterations = 5
reference_nll_break_condition = -float("inf")

approximate_kernel_diagonal_regularisation = 1e-10

reference_gp_empirical_risk_scheme = EmpiricalRiskScheme.cross_entropy
approximate_gp_empirical_risk_scheme = EmpiricalRiskScheme.cross_entropy
approximate_gp_regularisation_scheme = RegularisationScheme.multinomial_wasserstein
tempered_gp_empirical_risk_scheme = EmpiricalRiskScheme.negative_log_likelihood

reference_save_checkpoint_frequency = 1000
approximate_save_checkpoint_frequency = 1000
tempered_save_checkpoint_frequency = 1000

reference_gp_trainer_settings = TrainerSettings(
    key=0,
    optimiser_scheme=OptimiserScheme.adabelief,
    learning_rate=1e-4,
    number_of_epochs=10000,
    batch_size=500,
    batch_shuffle=True,
    batch_drop_last=False,
)
approximate_gp_trainer_settings = TrainerSettings(
    key=0,
    optimiser_scheme=OptimiserScheme.adabelief,
    learning_rate=1e-3,
    number_of_epochs=50000,
    batch_size=500,
    batch_shuffle=True,
    batch_drop_last=False,
)
tempered_gp_trainer_settings = TrainerSettings(
    key=0,
    optimiser_scheme=OptimiserScheme.adabelief,
    learning_rate=1e-3,
    number_of_epochs=10000,
    batch_size=500,
    batch_shuffle=True,
    batch_drop_last=False,
)

# Run experiment
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
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
)
plot_images(
    data_list=[experiment_data.train for experiment_data in experiment_data_list],
    reshape_function=lambda x: x.reshape(28, 28),
    save_path=os.path.join(output_directory, "train-images.png"),
)

nn_mean = CNN(number_of_outputs=number_of_labels)
nn_mean_parameters = nn_mean.init(
    jax.random.PRNGKey(0),
    experiment_data_list[0].train.x[:1, ...].reshape(-1, 28, 28, 1),
)
# nn_mean = MLP(
#     features=[100, 100, len(labels_to_include)],
# )
# nn_mean_parameters = nn_mean.init(
#     jax.random.PRNGKey(0), experiment_data_list[0].train.x[:1, ...]
# )
custom_mean = CustomMean(
    mean_function=lambda parameters, x: nn_mean.apply(parameters, x),
    number_output_dimensions=number_of_labels,
    preprocess_function=lambda x: x.reshape(-1, 28, 28, 1),
    # preprocess_function=lambda x: x.reshape(-1, 784),
)

nngp_kernel = MLPGPKernel(features=[100, 100, 1])
# nngp_kernel = CNNGPKernel(number_of_outputs=number_of_labels)
nngp_kernel_parameters = nngp_kernel.initialise_parameters()
single_label_kernel = CustomKernel(
    kernel_function=nngp_kernel,
    # preprocess_function=lambda x: x.reshape(-1, 28, 28, 1),
    preprocess_function=lambda x: x.reshape(-1, 784),
)
single_label_kernel_parameters = CustomKernel.Parameters.construct(
    custom=nngp_kernel_parameters,
)
(
    reference_gp,
    reference_gp_parameters,
    reference_post_epoch_histories,
    inducing_data_list,
) = meta_train_reference_gp(
    data_list=[experiment_data.train for experiment_data in experiment_data_list],
    empirical_risk_scheme=reference_gp_empirical_risk_scheme,
    trainer_settings=reference_gp_trainer_settings,
    kernel=MultiOutputKernel(
        kernels=[single_label_kernel for _ in range(number_of_labels)]
    ),
    kernel_parameters=MultiOutputKernel.Parameters.construct(
        kernels=[single_label_kernel_parameters for _ in range(number_of_labels)],
    ),
    number_of_inducing_per_label=number_of_inducing_per_label,
    number_of_iterations=reference_number_of_iterations,
    empirical_risk_break_condition=reference_nll_break_condition,
    save_checkpoint_frequency=reference_save_checkpoint_frequency,
    checkpoint_path=os.path.join(
        output_directory,
        checkpoints_folder_name,
        "reference-gp",
    ),
)
reference_accuracy = calculate_metric(
    gp=reference_gp,
    gp_parameters=reference_gp_parameters,
    data=reduce(
        operator.add, [experiment_data.test for experiment_data in experiment_data_list]
    ),
    metric_scheme=ClassificationMetricScheme.accuracy,
)
plot_images(
    data_list=inducing_data_list,
    reshape_function=lambda x: x.reshape(28, 28),
    save_path=os.path.join(output_directory, f"inducing-images.png"),
)
plot_losses(
    losses=[
        [x["empirical-risk"] for x in reference_post_epoch_history]
        for reference_post_epoch_history in reference_post_epoch_histories
    ],
    labels=[f"iteration-{i}" for i in range(reference_number_of_iterations)],
    loss_name=reference_gp_empirical_risk_scheme.value,
    title=f"Reference GP Empirical Risk: MNIST",
    save_path=os.path.join(
        output_directory,
        "reference-gp-losses.png",
    ),
)
print(
    f"Reference GP Accuracy:{reference_accuracy}",
)

approximate_experiment_directory = os.path.join(
    output_directory,
    approximate_gp_regularisation_scheme.value,
)
if not os.path.exists(approximate_experiment_directory):
    os.makedirs(approximate_experiment_directory)
svgp_kernels: List[KernelisedSVGPKernel] = []
svgp_kernel_parameters: List[KernelisedSVGPKernel.Parameters] = []
for i in range(number_of_labels):
    svgp_kernels.append(
        KernelisedSVGPKernel(
            reference_kernel=reference_gp.kernel.kernels[i],
            reference_kernel_parameters=reference_gp_parameters.kernel.kernels[i],
            log_observation_noise=reference_gp_parameters.log_observation_noise[i],
            inducing_points=inducing_data_list[i].x,
            training_points=experiment_data_list[i].train.x,
            diagonal_regularisation=approximate_kernel_diagonal_regularisation,
            base_kernel=reference_gp.kernel.kernels[i],
        )
    )
    svgp_kernel_parameters.append(
        KernelisedSVGPKernel.Parameters.construct(
            base_kernel=reference_gp_parameters.kernel.kernels[i],
        )
    )
approximate_gp = ApproximateGPClassification(
    mean=custom_mean,
    kernel=MultiOutputKernel(
        kernels=svgp_kernels,
    ),
)
initial_approximate_gp_parameters = ApproximateGPClassification.Parameters.construct(
    mean=CustomMean.Parameters.construct(
        custom=nn_mean_parameters,
    ),
    kernel=MultiOutputKernel.Parameters.construct(
        kernels=svgp_kernel_parameters,
    ),
)
approximate_gp_parameters, approximate_post_epoch_history = train_approximate_gp(
    data=reduce(
        operator.add,
        [experiment_data.train for experiment_data in experiment_data_list],
    ),
    empirical_risk_scheme=approximate_gp_empirical_risk_scheme,
    regularisation_scheme=approximate_gp_regularisation_scheme,
    trainer_settings=approximate_gp_trainer_settings,
    approximate_gp=approximate_gp,
    approximate_gp_parameters=initial_approximate_gp_parameters,
    regulariser=reference_gp,
    regulariser_parameters=reference_gp_parameters,
    save_checkpoint_frequency=approximate_save_checkpoint_frequency,
    checkpoint_path=os.path.join(
        output_directory,
        checkpoints_folder_name,
        "approximate-gp",
    ),
)
approximate_accuracy = calculate_metric(
    gp=approximate_gp,
    gp_parameters=approximate_gp_parameters,
    data=reduce(
        operator.add, [experiment_data.test for experiment_data in experiment_data_list]
    ),
    metric_scheme=ClassificationMetricScheme.accuracy,
)
plot_losses(
    losses=[x["gvi-objective"] for x in approximate_post_epoch_history],
    labels="gvi-objective",
    loss_name=f"{approximate_gp_empirical_risk_scheme.value}+{approximate_gp_regularisation_scheme.value}",
    title=" ".join(
        [
            f"Approximate GP Objective ({approximate_gp_regularisation_scheme.value}):",
            "MNIST",
        ]
    ),
    save_path=os.path.join(
        approximate_experiment_directory,
        "approximate-gp-loss.png",
    ),
)
plot_two_losses(
    loss1=[x["empirical-risk"] for x in approximate_post_epoch_history],
    loss1_name=approximate_gp_empirical_risk_scheme.value,
    loss2=[x["regularisation"] for x in approximate_post_epoch_history],
    loss2_name=approximate_gp_regularisation_scheme.value,
    title=" ".join(
        [
            f"Approximate GP Objective Breakdown ({approximate_gp_regularisation_scheme.value}):",
            "MNIST",
        ]
    ),
    save_path=os.path.join(
        approximate_experiment_directory,
        "approximate-gp-loss-breakdown.png",
    ),
)
print(
    f"Approximate GP Accuracy ({approximate_gp_regularisation_scheme.value}):",
    approximate_accuracy,
)

tempered_approximate_gp = type(approximate_gp)(
    mean=approximate_gp.mean,
    kernel=MultiOutputKernel(
        kernels=[
            TemperedKernel(
                base_kernel=approximate_gp.kernel.kernels[i],
                base_kernel_parameters=approximate_gp_parameters.kernel.kernels[i],
                number_output_dimensions=approximate_gp.kernel.kernels[
                    i
                ].number_output_dimensions,
            )
            for i in range(number_of_labels)
        ],
    ),
)
initial_tempered_gp_parameters = approximate_gp.Parameters.construct(
    log_observation_noise=approximate_gp_parameters.log_observation_noise,
    mean=approximate_gp_parameters.mean,
    kernel=MultiOutputKernel.Parameters.construct(
        kernels=[
            TemperedKernel.Parameters.construct(log_tempering_factor=jnp.log(1.0))
            for i in range(number_of_labels)
        ],
    ),
)
tempered_gp_parameters, tempered_post_epoch_history = train_tempered_gp(
    data=reduce(
        operator.add,
        [experiment_data.validation for experiment_data in experiment_data_list],
    ),
    empirical_risk_scheme=tempered_gp_empirical_risk_scheme,
    trainer_settings=tempered_gp_trainer_settings,
    tempered_gp=tempered_approximate_gp,
    tempered_gp_parameters=initial_tempered_gp_parameters,
    save_checkpoint_frequency=tempered_save_checkpoint_frequency,
    checkpoint_path=os.path.join(
        output_directory,
        checkpoints_folder_name,
        "tempered-gp",
        approximate_gp_regularisation_scheme.value,
    ),
)
plot_losses(
    losses=[x["empirical-risk"] for x in tempered_post_epoch_history],
    labels="empirical-risk",
    loss_name=tempered_gp_empirical_risk_scheme.value,
    title=" ".join(
        [
            f"Tempered Approximate GP Empirical Risk ({approximate_gp_regularisation_scheme.value}):",
            "MNIST",
        ]
    ),
    save_path=os.path.join(
        approximate_experiment_directory,
        "tempered-approximate-gp-loss.png",
    ),
)
tempered_accuracy = calculate_metric(
    gp=tempered_approximate_gp,
    gp_parameters=tempered_gp_parameters,
    data=reduce(
        operator.add, [experiment_data.test for experiment_data in experiment_data_list]
    ),
    metric_scheme=ClassificationMetricScheme.accuracy,
)
print(
    f"Tempered Approximate GP Accuracy ({approximate_gp_regularisation_scheme.value}):",
    tempered_accuracy,
)
