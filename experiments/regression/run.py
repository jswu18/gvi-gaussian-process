import os
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax
from flax.training import orbax_utils
from neural_tangents import stax

from experiments.data import ExperimentData, set_up_experiment
from experiments.neural_networks import MultiLayerPerceptron
from experiments.regression.plotters import (
    plot_losses,
    plot_regression,
    plot_two_losses,
)
from experiments.regression.toy_curves import CURVE_FUNCTIONS, Curve
from experiments.utils import train_gvi, train_nll, train_tempered_nll
from src import GeneralisedVariationalInference
from src.distributions import Gaussian
from src.empirical_risks import NegativeLogLikelihood
from src.gps import ApproximateGPRegression, GPRegression
from src.gps.base.approximate_base import ApproximateGPBase
from src.gps.base.base import GPBase, GPBaseParameters
from src.kernels import CustomKernel, TemperedKernel, TemperedKernelParameters
from src.kernels.approximate.svgp_kernel import StochasticVariationalKernel
from src.kernels.base import KernelBase, KernelBaseParameters
from src.means import ConstantMean, CustomMean
from src.regularisations import WassersteinRegularisation
from src.utils.custom_types import PRNGKey

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def run_reference_gp(
    key: PRNGKey,
    curve_function: Curve,
    experiment_data: ExperimentData,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
    lr: float,
    training_epochs: int,
    save_checkpoint_frequency: int,
    batch_size: int,
    load_checkpoint: bool,
    output_folder: str,
) -> Tuple[GPBase, GPBaseParameters]:
    gp = GPRegression(
        x=experiment_data.x_inducing,
        y=experiment_data.y_inducing,
        kernel=kernel,
        mean=ConstantMean(),
    )
    gp_parameters = gp.generate_parameters(
        {
            "log_observation_noise": jnp.log(1.0),
            "mean": {"constant": 0},
            "kernel": kernel_parameters.dict(),
        }
    )
    reference_parameters_path = os.path.join(output_folder, f"reference.ckpt")
    key, subkey = jax.random.split(key)
    if load_checkpoint:
        gp_parameters = gp.generate_parameters(
            orbax_checkpointer.restore(reference_parameters_path)
        )
    else:
        gp_parameters, reference_losses = train_nll(
            key=subkey,
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
        )
        ckpt = gp_parameters.dict()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            reference_parameters_path, ckpt, save_args=save_args, force=True
        )
        fig = plot_losses(
            losses=reference_losses,
            loss_name="Negative Log Likelihood",
            title=f"Reference GP NLL Loss ({curve_function.__name__})",
        )
        fig.savefig(
            os.path.join(output_folder, "reference-losses.png"), bbox_inches="tight"
        )
        plt.close(fig)
        np.save(
            os.path.join(output_folder, "reference-losses.npy"),
            np.array(reference_losses),
        )
    predicted_distribution = Gaussian(
        **gp.predict_probability(
            x=experiment_data.x,
            parameters=gp_parameters,
        ).dict()
    )
    fig = plot_regression(
        experiment_data=experiment_data,
        mean=predicted_distribution.mean,
        covariance=predicted_distribution.covariance,
        title=f"Reference GP ({curve_function.__name__})",
    )
    fig.savefig(os.path.join(output_folder, "reference.png"), bbox_inches="tight")
    plt.close(fig)
    return gp, gp_parameters


def run_approximate_gp(
    key: PRNGKey,
    gp: GPBase,
    gp_parameters: GPBaseParameters,
    experiment_data: ExperimentData,
    curve_function: Curve,
    lr: float,
    training_epochs: int,
    save_checkpoint_frequency: int,
    batch_size: int,
    load_checkpoint: bool,
    neural_network: nn.Module,
    diagonal_regularisation: float,
    include_eigendecomposition: bool,
    eigenvalue_regularisation: float,
    output_folder: str,
) -> Tuple[ApproximateGPBase, GPBaseParameters]:

    approximate_gp = ApproximateGPRegression(
        kernel=StochasticVariationalKernel(
            reference_kernel=gp.kernel,
            reference_kernel_parameters=gp_parameters.kernel,
            log_observation_noise=gp_parameters.log_observation_noise,
            inducing_points=experiment_data.x_inducing,
            training_points=experiment_data.x_train,
            diagonal_regularisation=diagonal_regularisation,
        ),
        mean=CustomMean(
            mean_function=lambda parameters, x: neural_network.apply(parameters, x),
        ),
    )
    key, subkey = jax.random.split(key)
    approximate_gp_parameters = approximate_gp.generate_parameters(
        {
            "mean": approximate_gp.mean.generate_parameters(
                {
                    "custom": neural_network.init(
                        subkey, experiment_data.x_train[:1, ...]
                    )
                }
            ),
            "kernel": approximate_gp.kernel.initialise_random_parameters(subkey),
        }
    )
    regularisation = WassersteinRegularisation(
        gp=approximate_gp,
        regulariser=gp,
        regulariser_parameters=gp_parameters,
        include_eigendecomposition=include_eigendecomposition,
        eigenvalue_regularisation=eigenvalue_regularisation,
    )
    empirical_risk = NegativeLogLikelihood(
        gp=approximate_gp,
    )
    gvi = GeneralisedVariationalInference(
        empirical_risk=empirical_risk,
        regularisation=regularisation,
    )

    key, subkey = jax.random.split(key)
    approximate_parameters_path = os.path.join(output_folder, f"approximate.ckpt")
    if load_checkpoint:
        approximate_gp_parameters = approximate_gp.generate_parameters(
            orbax_checkpointer.restore(approximate_parameters_path)
        )
    else:
        approximate_gp_parameters, gvi_losses, emp_risk_losses, reg_losses = train_gvi(
            key=subkey,
            gp_parameters=approximate_gp_parameters,
            gvi=gvi,
            x=experiment_data.x_train,
            y=experiment_data.y_train,
            learning_rate=lr,
            number_of_epochs=training_epochs,
            save_checkpoint_frequency=save_checkpoint_frequency,
            batch_size=batch_size,
            checkpoint_path=os.path.join(
                output_folder, "training-checkpoints", "approximate"
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
            title=f"GVI Loss ({curve_function.__name__})",
        )
        fig.savefig(
            os.path.join(output_folder, "approximate-gvi-losses.png"),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = plot_two_losses(
            loss1=emp_risk_losses,
            loss1_name="Empirical Risk",
            loss2=reg_losses,
            loss2_name="Regularisation",
            title=f"GVI Loss Decomposed ({curve_function.__name__})",
        )
        fig.savefig(
            os.path.join(output_folder, "approximate-gvi-losses-breakdown.png"),
            bbox_inches="tight",
        )
        plt.close(fig)
        np.save(
            os.path.join(output_folder, "approximate-gvi-losses.npy"),
            np.array(gvi_losses),
        )
        np.save(
            os.path.join(output_folder, "approximate-emp-risk-losses.npy"),
            np.array(emp_risk_losses),
        )
        np.save(
            os.path.join(output_folder, "approximate-reg-losses.npy"),
            np.array(reg_losses),
        )

    predicted_distribution = Gaussian(
        **approximate_gp.predict_probability(
            x=experiment_data.x,
            parameters=approximate_gp_parameters,
        ).dict()
    )
    fig = plot_regression(
        experiment_data=experiment_data,
        mean=predicted_distribution.mean,
        covariance=predicted_distribution.covariance,
        title=f"Approximate GP ({curve_function.__name__})",
    )
    fig.savefig(os.path.join(output_folder, "approximate.png"), bbox_inches="tight")
    plt.close(fig)
    return approximate_gp, approximate_gp_parameters


def run_tempered_gp(
    key: PRNGKey,
    gp: ApproximateGPBase,
    gp_parameters: GPBaseParameters,
    experiment_data: ExperimentData,
    curve_function: Curve,
    lr: float,
    training_epochs: int,
    save_checkpoint_frequency: int,
    batch_size: int,
    load_checkpoint: bool,
    output_folder: str,
) -> Tuple[GPBase, GPBaseParameters]:
    tempered_gp = type(gp)(
        mean=gp.mean,
        kernel=TemperedKernel(
            base_kernel=gp.kernel,
            base_kernel_parameters=gp_parameters.kernel,
            number_output_dimensions=gp.kernel.number_output_dimensions,
        ),
    )
    tempered_gp_parameters = tempered_gp.Parameters(
        log_observation_noise=gp_parameters.log_observation_noise,
        mean=gp_parameters.mean,
        kernel=TemperedKernelParameters(log_tempering_factor=jnp.log(2.0)),
    )

    parameters_path = os.path.join(output_folder, f"tempered.ckpt")
    key, subkey = jax.random.split(key)
    if load_checkpoint:
        tempered_gp_parameters = gp.generate_parameters(
            orbax_checkpointer.restore(parameters_path)
        )
    else:
        tempered_gp_parameters, losses = train_tempered_nll(
            key=subkey,
            gp=tempered_gp,
            gp_parameters=tempered_gp_parameters,
            base_gp_parameters=gp_parameters,
            x=experiment_data.x_validation,
            y=experiment_data.y_validation,
            learning_rate=lr,
            number_of_epochs=training_epochs,
            save_checkpoint_frequency=save_checkpoint_frequency,
            batch_size=batch_size,
            checkpoint_path=os.path.join(
                output_folder, "training-checkpoints", "tempered"
            ),
        )
        ckpt = gp_parameters.dict()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(parameters_path, ckpt, save_args=save_args, force=True)
        fig = plot_losses(
            losses=losses,
            loss_name="Negative Log Likelihood",
            title=f"Tempered Approximate GP NLL Loss ({curve_function.__name__})",
        )
        fig.savefig(
            os.path.join(output_folder, "tempered-losses.png"), bbox_inches="tight"
        )
        plt.close(fig)
        np.save(os.path.join(output_folder, "tempered-losses.npy"), np.array(losses))

    predicted_distribution = Gaussian(
        **tempered_gp.predict_probability(
            x=experiment_data.x,
            parameters=tempered_gp_parameters,
        ).dict()
    )
    fig = plot_regression(
        experiment_data=experiment_data,
        mean=predicted_distribution.mean,
        covariance=predicted_distribution.covariance,
        title=f"Tempered Approximate GP ({curve_function.__name__})",
    )
    fig.savefig(os.path.join(output_folder, "tempered.png"), bbox_inches="tight")
    plt.close(fig)
    return tempered_gp, tempered_gp_parameters


def run_experiment(
    key: PRNGKey,
    curve_function: Curve,
    x: jnp.ndarray,
    sigma_true: float,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    number_of_inducing_points: int,
    train_data_percentage: float,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
    reference_gp_lr: float,
    reference_gp_training_epochs: int,
    reference_save_checkpoint_frequency: int,
    reference_gp_batch_size: int,
    reference_load_checkpoint: bool,
    approximate_gp_lr: float,
    approximate_gp_training_epochs: int,
    approximate_save_checkpoint_frequency: int,
    approximate_gp_batch_size: int,
    approximate_load_checkpoint: bool,
    output_directory: str,
    neural_network: nn.Module,
    diagonal_regularisation: float,
    include_eigendecomposition: bool,
    eigenvalue_regularisation: float,
    tempered_gp_lr: float,
    tempered_gp_training_epochs: int,
    tempered_save_checkpoint_frequency: int,
    tempered_gp_batch_size: int,
    tempered_load_checkpoint: bool,
):
    curve_name = type(curve_function).__name__.lower()
    output_folder = os.path.join(output_directory, curve_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    key, subkey = jax.random.split(key)
    experiment_data = set_up_experiment(
        key=subkey,
        curve_function=curve_function,
        x=x,
        sigma_true=sigma_true,
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
        number_of_inducing_points=number_of_inducing_points,
        train_data_percentage=train_data_percentage,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
    )
    fig = plot_regression(
        experiment_data=experiment_data,
        title=f"{curve_function.__name__}",
    )
    fig.savefig(os.path.join(output_folder, f"{curve_name}.png"), bbox_inches="tight")
    plt.close(fig)
    key, subkey = jax.random.split(key)
    gp, gp_parameters = run_reference_gp(
        key=subkey,
        curve_function=curve_function,
        experiment_data=experiment_data,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
        lr=reference_gp_lr,
        training_epochs=reference_gp_training_epochs,
        save_checkpoint_frequency=reference_save_checkpoint_frequency,
        batch_size=reference_gp_batch_size,
        load_checkpoint=reference_load_checkpoint,
        output_folder=output_folder,
    )
    key, subkey = jax.random.split(key)
    approximate_gp, approximate_gp_parameters = run_approximate_gp(
        key=subkey,
        gp=gp,
        gp_parameters=gp_parameters,
        experiment_data=experiment_data,
        curve_function=curve_function,
        lr=approximate_gp_lr,
        training_epochs=approximate_gp_training_epochs,
        save_checkpoint_frequency=approximate_save_checkpoint_frequency,
        batch_size=approximate_gp_batch_size,
        load_checkpoint=approximate_load_checkpoint,
        neural_network=neural_network,
        diagonal_regularisation=diagonal_regularisation,
        include_eigendecomposition=include_eigendecomposition,
        eigenvalue_regularisation=eigenvalue_regularisation,
        output_folder=output_folder,
    )
    key, subkey = jax.random.split(key)
    run_tempered_gp(
        key=subkey,
        gp=approximate_gp,
        gp_parameters=approximate_gp_parameters,
        experiment_data=experiment_data,
        curve_function=curve_function,
        lr=tempered_gp_lr,
        training_epochs=tempered_gp_training_epochs,
        save_checkpoint_frequency=tempered_save_checkpoint_frequency,
        batch_size=tempered_gp_batch_size,
        load_checkpoint=tempered_load_checkpoint,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    NUMBER_OF_DATA_POINTS = 500
    SIGMA_TRUE = 0.5
    TRAIN_DATA_PERCENTAGE = 0.8
    NUMBER_OF_TEST_INTERVALS = 2
    TOTAL_NUMBER_OF_INTERVALS = 8
    NUMBER_OF_INDUCING_POINTS = int(2 * np.sqrt(NUMBER_OF_DATA_POINTS))
    REFERENCE_GP_LR = 1e-3
    REFERENCE_GP_TRAINING_EPOCHS = 5000
    REFERENCE_SAVE_CHECKPOINT_FREQUENCY = 1000
    REFERENCE_GP_BATCH_SIZE = 100
    REFERENCE_LOAD_CHECKPOINT = False
    OUTPUT_DIRECTORY = "outputs"
    DIAGONAL_REGULARISATION = 1e-10
    INCLUDE_EIGENDECOMPOSITION = False
    EIGENVALUE_REGULARISATION = 1e-10
    APPROXIMATE_GP_LR = 5e-3
    APPROXIMATE_GP_TRAINING_EPOCHS = 50000
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
            diagonal_regularisation=DIAGONAL_REGULARISATION,
            include_eigendecomposition=INCLUDE_EIGENDECOMPOSITION,
            eigenvalue_regularisation=EIGENVALUE_REGULARISATION,
            tempered_gp_lr=TEMPERED_GP_LR,
            tempered_gp_training_epochs=TEMPERED_GP_TRAINING_EPOCHS,
            tempered_save_checkpoint_frequency=TEMPERED_SAVE_CHECKPOINT_FREQUENCY,
            tempered_gp_batch_size=TEMPERED_GP_BATCH_SIZE,
            tempered_load_checkpoint=TEMPERED_LOAD_CHECKPOINT,
        )
