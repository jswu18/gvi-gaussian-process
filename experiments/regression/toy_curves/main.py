import os
from typing import List, Tuple, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax
from flax.training import orbax_utils
from neural_tangents import stax

from experiments import schemes
from experiments.data import ExperimentData
from experiments.nn_means import MultiLayerPerceptron
from experiments.nngp_kernels import MultiLayerPerceptronKernel
from experiments.plotters import plot_losses, plot_two_losses
from experiments.regression import data, runners
from experiments.regression.plotters import plot_regression
from experiments.regression.toy_curves.curves import CURVE_FUNCTIONS, Curve
from experiments.trainer import TrainerSettings
from experiments.utils import calculate_inducing_points
from src import GeneralisedVariationalInference
from src.distributions import Gaussian
from src.empirical_risks import NegativeLogLikelihood
from src.gps import ApproximateGPRegression, GPRegression
from src.gps.base.approximate_base import ApproximateGPBase
from src.gps.base.base import GPBase, GPBaseParameters
from src.kernels import (
    CustomKernel,
    CustomKernelParameters,
    NeuralNetworkKernel,
    TemperedKernel,
    TemperedKernelParameters,
)
from src.kernels.approximate.generalised_svgp_kernel import (
    GeneralisedStochasticVariationalKernel,
)
from src.kernels.approximate.svgp_diagonal_kernel import StochasticVariationalKernel
from src.kernels.base import KernelBase, KernelBaseParameters
from src.kernels.non_stationary import PolynomialKernel
from src.kernels.standard import ARDKernel, ARDKernelParameters
from src.means import ConstantMean, CustomMean
from src.regularisations import (
    SquaredDifferenceRegularisation,
    WassersteinRegularisation,
)
from src.regularisations.base import RegularisationBase
from src.regularisations.point_wise import (
    PointWiseBhattacharyyaRegularisation,
    PointWiseHellingerRegularisation,
    PointWiseKLRegularisation,
    PointWiseRenyiRegularisation,
    PointWiseSymmetricKLRegularisation,
    PointWiseWassersteinRegularisation,
)
from src.utils.custom_types import PRNGKey

# orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

jax.config.update("jax_enable_x64", True)
# NUMBER_OF_DATA_POINTS = 1000
# SIGMA_TRUE = 0.5
# NUMBER_OF_INDUCING_POINTS = int(np.sqrt(NUMBER_OF_DATA_POINTS))
# REFERENCE_GP_LR = 5e-5
# REFERENCE_GP_TRAINING_EPOCHS = 20000
# REFERENCE_SAVE_CHECKPOINT_FREQUENCY = 1000
# REFERENCE_GP_BATCH_SIZE = 1000
# REFERENCE_LOAD_CHECKPOINT = False
# DIAGONAL_REGULARISATION = 1e-10
# INCLUDE_EIGENDECOMPOSITION = False
# EIGENVALUE_REGULARISATION = 1e-10
# APPROXIMATE_GP_LR = 1e-3
# APPROXIMATE_GP_TRAINING_EPOCHS = 20000
# APPROXIMATE_SAVE_CHECKPOINT_FREQUENCY = 1000
# APPROXIMATE_GP_BATCH_SIZE = 1000
# APPROXIMATE_LOAD_CHECKPOINT = False
# TEMPERED_GP_LR = 1e-3
# TEMPERED_GP_TRAINING_EPOCHS = 2000
# TEMPERED_SAVE_CHECKPOINT_FREQUENCY = 1000
# TEMPERED_GP_BATCH_SIZE = 1000
# TEMPERED_LOAD_CHECKPOINT = False
# NLL_BREAK_CONDITION = -10


output_directory = "outputs"


number_of_data_points = 1000
x = jnp.linspace(-2, 2, number_of_data_points, dtype=np.float64).reshape(-1, 1)
number_of_test_intervals = 5
total_number_of_intervals = 20
train_data_percentage = 0.8
sigma_true = 0.5

reference_gp_empirical_risk_scheme = schemes.EmpiricalRisk.negative_log_likelihood
reference_gp_trainer_settings = TrainerSettings(
    key=0,
    optimiser_scheme=schemes.Optimiser.adabeleif,
    learning_rate=5e-5,
    number_of_epochs=20000,
    batch_size=1000,
    batch_shuffle=True,
    batch_drop_last=False,
)
number_of_inducing_points = int(np.sqrt(number_of_data_points))
reference_save_checkpoint_frequency = 1000
reference_number_of_iterations = 5
reference_nll_break_condition = -10

nn_architecture = [10, 1]

nngp_kernel = MultiLayerPerceptronKernel(
    features=nn_architecture,
)
nn_mean = MultiLayerPerceptron(
    features=nn_architecture,
)
for curve_function in CURVE_FUNCTIONS:
    experiment_data = data.set_up_regression_experiment(
        key=jax.random.PRNGKey(curve_function.seed),
        x=x,
        y=curve_function(
            key=jax.random.PRNGKey(curve_function.seed),
            x=x,
            sigma_true=sigma_true,
        ),
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
        train_data_percentage=train_data_percentage,
    )
    runners.meta_train_reference_gp(
        data=experiment_data.train,
        empirical_risk_scheme=reference_gp_empirical_risk_scheme,
        trainer_settings=reference_gp_trainer_settings,
        kernel=CustomKernel(
            kernel_function=nngp_kernel,
        ),
        kernel_parameters=CustomKernelParameters(
            custom=nngp_kernel.initialise_parameters().dict(),
        ),
        number_of_inducing_points=number_of_inducing_points,
        save_checkpoint_frequency=reference_save_checkpoint_frequency,
        checkpoint_path=f"{output_directory}/reference_gp/{curve_function.__name__.lower()}",
        number_of_iterations=reference_number_of_iterations,
        nll_break_condition=reference_nll_break_condition,
    )
