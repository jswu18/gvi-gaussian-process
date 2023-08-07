from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp

from experiments import resolvers, schemes
from experiments.data import Data
from experiments.trainer import Trainer, TrainerSettings
from experiments.utils import calculate_inducing_points
from src import GeneralisedVariationalInference
from src.gps import GPRegression, GPRegressionParameters
from src.kernels.base import KernelBase, KernelBaseParameters
from src.means import ConstantMean


def train_reference_gp(
    data: Data,
    empirical_risk_scheme: schemes.EmpiricalRisk,
    trainer_settings: TrainerSettings,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
    number_of_inducing_points: int,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
    nll_break_condition: float,
) -> Tuple[GPRegression, GPRegressionParameters, Data, List[Dict[str, float]]]:
    data_inducing = calculate_inducing_points(
        key=jax.random.PRNGKey(trainer_settings.key),
        data=data,
        number_of_inducing_points=number_of_inducing_points,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
    )
    gp = GPRegression(
        x=data_inducing.x,
        y=data_inducing.y,
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
    empirical_risk = resolvers.empirical_risk(
        empirical_risk_scheme=empirical_risk_scheme,
        gp=gp,
    )
    trainer = Trainer(
        save_checkpoint_frequency=save_checkpoint_frequency,
        checkpoint_path=checkpoint_path,
        post_epoch_callback=lambda parameters: {
            "negative-log-likelihood": empirical_risk.calculate_empirical_risk(
                parameters, data.x, data.y
            )
        },
        break_condition_function=lambda parameters: empirical_risk.calculate_empirical_risk(
            parameters, data.x, data.y
        )
        < nll_break_condition,
    )
    gp_parameters, post_epoch_history = trainer.train(
        trainer_settings=trainer_settings,
        parameters=gp_parameters,
        data=data_inducing,
        loss_function=lambda parameters: empirical_risk.calculate_empirical_risk(
            parameters, data.x, data.y
        ),
    )
    gp_parameters = gp.generate_parameters(gp_parameters.dict())
    return gp, gp_parameters, data_inducing, post_epoch_history


def meta_train_reference_gp(
    data: Data,
    empirical_risk_scheme: schemes.EmpiricalRisk,
    trainer_settings: TrainerSettings,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
    number_of_inducing_points: int,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
    number_of_iterations: int,
    nll_break_condition: float = -float("inf"),
) -> Tuple[GPRegression, GPRegressionParameters, Data]:
    gp, gp_parameters, data_inducing, post_epoch_history = train_reference_gp(
        data=data,
        empirical_risk_scheme=empirical_risk_scheme,
        trainer_settings=trainer_settings,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
        number_of_inducing_points=number_of_inducing_points,
        save_checkpoint_frequency=save_checkpoint_frequency,
        checkpoint_path=checkpoint_path,
        nll_break_condition=nll_break_condition,
    )
    kernel_parameters = gp_parameters.kernel
    for i in range(number_of_iterations - 1):
        gp, gp_parameters, data_inducing, post_epoch_history = train_reference_gp(
            data=data,
            empirical_risk_scheme=empirical_risk_scheme,
            trainer_settings=trainer_settings,
            kernel=kernel,
            kernel_parameters=kernel_parameters,
            number_of_inducing_points=number_of_inducing_points,
            save_checkpoint_frequency=save_checkpoint_frequency,
            checkpoint_path=checkpoint_path,
            nll_break_condition=nll_break_condition,
        )
        kernel_parameters = gp_parameters.kernel
    return gp, gp_parameters, data_inducing


def train_approximate_gp(
    data: Data,
    empirical_risk_scheme: schemes.EmpiricalRisk,
    regularisation_scheme: schemes.Regularisation,
    trainer_settings: TrainerSettings,
    approximate_gp: GPRegression,
    approximate_gp_parameters: GPRegressionParameters,
    regulariser: GPRegression,
    regulariser_parameters: GPRegressionParameters,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
) -> Tuple[GPRegressionParameters, List[Dict[str, float]]]:
    empirical_risk = resolvers.empirical_risk(
        empirical_risk_scheme=empirical_risk_scheme,
        gp=approximate_gp,
    )
    regularisation = resolvers.regularisation(
        regularisation_scheme=regularisation_scheme,
        gp=approximate_gp,
        regulariser=regulariser,
        regulariser_parameters=regulariser_parameters,
    )
    gvi = GeneralisedVariationalInference(
        empirical_risk=empirical_risk,
        regularisation=regularisation,
    )
    trainer = Trainer(
        save_checkpoint_frequency=save_checkpoint_frequency,
        checkpoint_path=checkpoint_path,
        post_epoch_callback=lambda parameters: {
            "negative-log-likelihood": empirical_risk.calculate_empirical_risk(
                parameters, data.x, data.y
            ),
            "regularisation": regularisation.calculate_regularisation(parameters, x),
            "gvi-objective": gvi.calculate_loss(parameters, data.x, data.y),
        },
    )
    gp_parameters, post_epoch_history = trainer.train(
        trainer_settings=trainer_settings,
        parameters=approximate_gp_parameters,
        data=data,
        loss_function=lambda parameters: gvi.calculate_loss(parameters, data.x, data.y),
    )
    return approximate_gp.generate_parameters(gp_parameters.dict()), post_epoch_history


def train_tempered_gp(
    data: Data,
    empirical_risk_scheme: schemes.EmpiricalRisk,
    trainer_settings: TrainerSettings,
    tempered_gp: GPRegression,
    tempered_gp_parameters: GPRegressionParameters,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
) -> Tuple[GPRegressionParameters, List[Dict[str, float]]]:
    empirical_risk = resolvers.empirical_risk(
        empirical_risk_scheme=empirical_risk_scheme,
        gp=tempered_gp,
    )
    trainer = Trainer(
        save_checkpoint_frequency=save_checkpoint_frequency,
        checkpoint_path=checkpoint_path,
        post_epoch_callback=lambda parameters: {
            "negative-log-likelihood": empirical_risk.calculate_empirical_risk(
                parameters=tempered_gp.Parameters(
                    log_observation_noise=tempered_gp_parameters.log_observation_noise,
                    mean=tempered_gp_parameters.mean,
                    kernel=parameters,
                ),
                x=data.x,
                y=data.y,
            ),
        },
    )
    tempered_kernel_parameters, post_epoch_history = trainer.train(
        trainer_settings=trainer_settings,
        parameters=tempered_gp_parameters.kernel,
        data=data,
        loss_function=lambda parameters, x, y: empirical_risk.calculate_empirical_risk(
            parameters=tempered_gp.Parameters(
                log_observation_noise=tempered_gp_parameters.log_observation_noise,
                mean=tempered_gp_parameters.mean,
                kernel=parameters,
            ),
            x=x,
            y=y,
        ),
    )
    return (
        tempered_gp.Parameters(
            log_observation_noise=tempered_gp_parameters.log_observation_noise,
            mean=tempered_gp_parameters.mean,
            kernel=tempered_kernel_parameters,
        ),
        post_epoch_history,
    )
