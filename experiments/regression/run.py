import os
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
from flax.training import orbax_utils
from neural_tangents import stax
from tqdm import tqdm

from experiments.examples.regression.plotters import plot_losses, plot_regression
from experiments.examples.regression.toy_regression_curves import CURVE_FUNCTIONS, Curve
from experiments.examples.regression.utils import (
    split_train_test_data,
    split_train_test_data_intervals,
)
from src.distributions import Gaussian
from src.empirical_risks import NegativeLogLikelihood
from src.gps import GPRegression
from src.gps.base.base import GPBase, GPBaseParameters
from src.inducing_points_selection import ConditionalVarianceInducingPointsSelector
from src.kernels import CustomKernel
from src.kernels.base import KernelBase, KernelBaseParameters
from src.means import ConstantMean
from src.utils.data import generate_batch

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

PRNGKey = Any  # pylint: disable=invalid-name


def split_train_test_validation_data(
    key: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    train_data_percentage: float,
):
    key, subkey = jax.random.split(key)
    (
        x_train_validation,
        y_train_validation,
        x_test,
        y_test,
    ) = split_train_test_data_intervals(
        subkey=subkey,
        x=x,
        y=y,
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
    )
    key, subkey = jax.random.split(key)
    x_train, y_train, x_validation, y_validation = split_train_test_data(
        key=subkey,
        x=x_train_validation,
        y=y_train_validation,
        train_data_percentage=train_data_percentage,
    )
    return x_train, y_train, x_test, y_test, x_validation, y_validation


def calculate_inducing_points(
    key: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    number_of_inducing_points: int,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    inducing_points_selector = ConditionalVarianceInducingPointsSelector()
    x_inducing, inducing_indices = inducing_points_selector.compute_inducing_points(
        key=key,
        training_inputs=x.reshape(-1, 1),
        number_of_inducing_points=number_of_inducing_points,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
    )
    y_inducing = y[inducing_indices]
    return x_inducing, y_inducing


def train_nll(
    key: PRNGKey,
    gp: GPBase,
    gp_parameters: GPBaseParameters,
    x: jnp.ndarray,
    y: jnp.ndarray,
    learning_rate: float,
    number_of_epochs: int,
    batch_size: int,
    checkpoint_path: str,
) -> Tuple[GPBaseParameters, List[float]]:
    optimizer = optax.adam(learning_rate)
    losses = []
    opt_state = optimizer.init(gp_parameters.dict())
    nll_loss = NegativeLogLikelihood(gp=gp)
    for epoch in tqdm(range(number_of_epochs)):
        losses.append(
            nll_loss.calculate_empirical_risk(
                parameters=gp_parameters,
                x=x,
                y=y,
            )
        )
        ckpt = gp_parameters.dict()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            os.path.join(checkpoint_path, f"epoch-{epoch}.ckpt"),
            ckpt,
            save_args=save_args,
            force=True,
        )
        key, subkey = jax.random.split(key)
        batch_generator = generate_batch(
            key=subkey,
            data=(x, y),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        data_batch = next(batch_generator, None)
        while data_batch is not None:
            x_batch, y_batch = data_batch
            gradients = jax.grad(
                lambda gp_parameters_dict: nll_loss.calculate_empirical_risk(
                    parameters=gp_parameters_dict,
                    x=x_batch,
                    y=y_batch,
                )
            )(gp_parameters.dict())
            updates, opt_state = optimizer.update(gradients, opt_state)
            gp_parameters = gp.generate_parameters(
                optax.apply_updates(gp_parameters.dict(), updates)
            )
            data_batch = next(batch_generator, None)
    losses.append(
        nll_loss.calculate_empirical_risk(
            parameters=gp_parameters,
            x=x,
            y=y,
        )
    )
    return gp_parameters, losses


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
    reference_gp_batch_size: int,
    reference_load_checkpoint: bool,
    output_directory: str,
):
    curve_name = type(curve_function).__name__.lower()
    output_folder = os.path.join(output_directory, curve_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    key, subkey = jax.random.split(key)
    y = curve_function(key=key, x=x, sigma_true=sigma_true)
    key, subkey = jax.random.split(key)
    (
        x_train,
        y_train,
        x_test,
        y_test,
        x_validation,
        y_validation,
    ) = split_train_test_validation_data(
        key=subkey,
        x=x,
        y=y,
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
        train_data_percentage=train_data_percentage,
    )
    key, subkey = jax.random.split(key)
    x_inducing, y_inducing = calculate_inducing_points(
        key=subkey,
        x=x_train,
        y=y_train,
        number_of_inducing_points=number_of_inducing_points,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
    )
    fig = plot_regression(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        x_inducing=x_inducing,
        y_inducing=y_inducing,
        title=f"{curve_function.__name__}",
    )
    fig.savefig(os.path.join(output_folder, f"{curve_name}.png"))
    gp = GPRegression(
        x=x_inducing,
        y=y_inducing,
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
    if reference_load_checkpoint:
        gp_parameters = gp.generate_parameters(
            orbax_checkpointer.restore(reference_parameters_path)
        )
    else:
        gp_parameters, reference_losses = train_nll(
            key=key,
            gp=gp,
            gp_parameters=gp_parameters,
            x=x_inducing,
            y=y_inducing,
            learning_rate=reference_gp_lr,
            number_of_epochs=reference_gp_training_epochs,
            batch_size=reference_gp_batch_size,
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
            title=f"{curve_function.__name__}",
        )
        fig.savefig(os.path.join(output_folder, "reference-losses.png"))
    predicted_distribution = Gaussian(
        **gp.predict_probability(
            x=x,
            parameters=gp_parameters,
        ).dict()
    )
    fig = plot_regression(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        x_inducing=x_inducing,
        y_inducing=y_inducing,
        x=x,
        mean=predicted_distribution.mean,
        covariance=predicted_distribution.covariance,
        title=f"{curve_function.__name__}",
    )
    fig.savefig(os.path.join(output_folder, "reference.png"))


if __name__ == "__main__":
    SEED = 0
    NUMBER_OF_DATA_POINTS = 1000
    SIGMA_TRUE = 0.5
    TRAIN_DATA_PERCENTAGE = 0.8
    NUMBER_OF_TEST_INTERVALS = 2
    TOTAL_NUMBER_OF_INTERVALS = 8
    NUMBER_OF_INDUCING_POINTS = int(np.sqrt(NUMBER_OF_DATA_POINTS))
    REFERENCE_GP_LR = 1e-3
    REFERENCE_GP_TRAINING_EPOCHS = 5000
    REFERENCE_GP_BATCH_SIZE = 100
    REFERENCE_LOAD_CHECKPOINT = True
    OUTPUT_DIRECTORY = "outputs"

    _, _, kernel_fn = stax.serial(
        stax.Dense(10, W_std=10, b_std=10),
        stax.Erf(),
        stax.Dense(1, W_std=10, b_std=10),
    )
    KERNEL = CustomKernel(lambda x1, x2: kernel_fn(x1, x2, "nngp"))
    KERNEL_PARAMETERS = KERNEL.Parameters()

    np.random.seed(SEED)
    KEY = jax.random.PRNGKey(SEED)
    X = jnp.linspace(-2, 2, NUMBER_OF_DATA_POINTS, dtype=np.float64).reshape(-1)

    for CURVE_FUNCTION in CURVE_FUNCTIONS:
        KEY, SUBKEY = jax.random.split(KEY)
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
            reference_gp_batch_size=REFERENCE_GP_BATCH_SIZE,
            reference_load_checkpoint=REFERENCE_LOAD_CHECKPOINT,
            output_directory=OUTPUT_DIRECTORY,
        )
