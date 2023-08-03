import os
from typing import List, Tuple

import jax
import optax
import orbax
from flax.training import orbax_utils
from jax import numpy as jnp
from tqdm import tqdm

from src import GeneralisedVariationalInference
from src.empirical_risks import NegativeLogLikelihood
from src.gps.base.base import GPBase, GPBaseParameters
from src.inducing_points_selection import ConditionalVarianceInducingPointsSelector
from src.kernels import TemperedKernelParameters
from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import PRNGKey
from src.utils.data import generate_batch

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


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
        training_inputs=jnp.atleast_2d(x).reshape(x.shape[0], -1),
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
    save_checkpoint_frequency: int,
    batch_size: int,
    checkpoint_path: str,
    nll_break_condition: float,
) -> Tuple[GPBaseParameters, List[float]]:
    optimizer = optax.adam(learning_rate)
    losses = []
    opt_state = optimizer.init(gp_parameters.dict())
    nll_loss = NegativeLogLikelihood(gp=gp)
    for epoch in tqdm(range(number_of_epochs)):
        if epoch % save_checkpoint_frequency == 0:
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
        if losses[-1] < nll_break_condition:
            break
    return gp_parameters, losses


def train_tempered_nll(
    key: PRNGKey,
    gp: GPBase,
    gp_parameters: GPBaseParameters,
    base_gp_parameters: GPBaseParameters,
    x: jnp.ndarray,
    y: jnp.ndarray,
    learning_rate: float,
    number_of_epochs: int,
    save_checkpoint_frequency: int,
    batch_size: int,
    checkpoint_path: str,
) -> Tuple[GPBaseParameters, List[float]]:
    optimizer = optax.adam(learning_rate)
    losses = []
    opt_state = optimizer.init(gp_parameters.kernel.dict())
    nll_loss = NegativeLogLikelihood(gp=gp)
    for epoch in tqdm(range(number_of_epochs)):
        if epoch % save_checkpoint_frequency == 0:
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
                lambda tempered_gp_parameters_kernel_dict: nll_loss.calculate_empirical_risk(
                    parameters=gp.Parameters(
                        log_observation_noise=base_gp_parameters.log_observation_noise,
                        mean=base_gp_parameters.mean,
                        kernel=TemperedKernelParameters(
                            **tempered_gp_parameters_kernel_dict
                        ),
                    ).dict(),
                    x=x_batch,
                    y=y_batch,
                )
            )(gp_parameters.kernel.dict())
            updates, opt_state = optimizer.update(gradients, opt_state)
            tempered_gp_kernel_parameters = gp.kernel.generate_parameters(
                optax.apply_updates(gp_parameters.kernel.dict(), updates)
            )
            gp_parameters = gp.Parameters(
                log_observation_noise=base_gp_parameters.log_observation_noise,
                mean=base_gp_parameters.mean,
                kernel=tempered_gp_kernel_parameters,
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


def train_gvi(
    key: PRNGKey,
    gp_parameters: GPBaseParameters,
    gvi: GeneralisedVariationalInference,
    x: jnp.ndarray,
    y: jnp.ndarray,
    learning_rate: float,
    number_of_epochs: int,
    save_checkpoint_frequency: int,
    batch_size: int,
    checkpoint_path: str,
) -> Tuple[GPBaseParameters, List[float], List[float], List[float]]:
    optimizer = optax.adam(learning_rate)
    gvi_losses = []
    emp_risk_losses = []
    reg_losses = []
    opt_state = optimizer.init(gp_parameters.dict())
    for epoch in tqdm(range(number_of_epochs)):
        if epoch % save_checkpoint_frequency == 0:
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
                lambda gp_parameters_dict: gvi.calculate_loss(
                    parameters=gp_parameters_dict,
                    x=x_batch,
                    y=y_batch,
                )
            )(gp_parameters.dict())
            updates, opt_state = optimizer.update(gradients, opt_state)
            gp_parameters = gvi.regularisation.gp.generate_parameters(
                optax.apply_updates(gp_parameters.dict(), updates)
            )
            data_batch = next(batch_generator, None)
        gvi_losses.append(
            gvi.calculate_loss(
                parameters=gp_parameters,
                x=x,
                y=y,
            )
        )
        emp_risk_losses.append(
            gvi.empirical_risk.calculate_empirical_risk(
                parameters=gp_parameters,
                x=x,
                y=y,
            )
        )
        reg_losses.append(
            gvi.regularisation.calculate_regularisation(
                parameters=gp_parameters,
                x=x,
            )
        )
    return gp_parameters, gvi_losses, emp_risk_losses, reg_losses
