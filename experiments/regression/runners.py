import os
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from tqdm import tqdm

from experiments import resolvers, schemes
from experiments.data import Data
from experiments.regression import plotters
from experiments.trainer import Trainer, TrainerSettings
from experiments.utils import calculate_inducing_points
from src import GeneralisedVariationalInference
from src.distributions import Gaussian
from src.gps import (
    ApproximateGPRegression,
    ApproximateGPRegressionParameters,
    GPRegression,
    GPRegressionParameters,
)
from src.gps.base.base import GPBaseParameters
from src.gps.base.regression_base import GPRegressionBase
from src.kernels.base import KernelBase, KernelBaseParameters
from src.means import ConstantMean


def train_reference_gp(
    data: Data,
    empirical_risk_scheme: schemes.EmpiricalRisk,
    trainer_settings: TrainerSettings,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
    number_of_inducing_points: int,
    empirical_risk_break_condition: float,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
) -> Tuple[GPRegression, GPRegressionParameters, List[Dict[str, float]]]:
    inducing_data = calculate_inducing_points(
        key=jax.random.PRNGKey(trainer_settings.key),
        data=data,
        number_of_inducing_points=number_of_inducing_points,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
    )
    gp = GPRegression(
        x=inducing_data.x,
        y=inducing_data.y,
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
            "empirical-risk": empirical_risk.calculate_empirical_risk(
                parameters, inducing_data.x, inducing_data.y
            )
        },
        break_condition_function=(
            lambda parameters: empirical_risk.calculate_empirical_risk(
                parameters, inducing_data.x, inducing_data.y
            )
            < empirical_risk_break_condition
        ),
    )
    gp_parameters, post_epoch_history = trainer.train(
        trainer_settings=trainer_settings,
        parameters=gp_parameters,
        data=inducing_data,
        loss_function=lambda parameters_dict, x, y: empirical_risk.calculate_empirical_risk(
            parameters=parameters_dict,
            x=x,
            y=y,
        ),
        disable_tqdm=True,
    )
    gp_parameters = gp.generate_parameters(gp_parameters.dict())
    return gp, gp_parameters, post_epoch_history


def meta_train_reference_gp(
    data: Data,
    empirical_risk_scheme: schemes.EmpiricalRisk,
    trainer_settings: TrainerSettings,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
    number_of_inducing_points: int,
    number_of_iterations: int,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
    empirical_risk_break_condition: float = -float("inf"),
) -> Tuple[GPRegression, GPRegressionParameters, List[List[Dict[str, float]]]]:
    post_epoch_histories = []
    gp, gp_parameters = None, None
    for i in tqdm(range(number_of_iterations)):
        gp, gp_parameters, post_epoch_history = train_reference_gp(
            data=data,
            empirical_risk_scheme=empirical_risk_scheme,
            trainer_settings=trainer_settings,
            kernel=kernel,
            kernel_parameters=kernel_parameters,
            number_of_inducing_points=number_of_inducing_points,
            save_checkpoint_frequency=save_checkpoint_frequency,
            checkpoint_path=os.path.join(checkpoint_path, f"iteration-{i}"),
            empirical_risk_break_condition=empirical_risk_break_condition,
        )
        kernel_parameters = gp_parameters.kernel
        post_epoch_histories.append(post_epoch_history)
        prediction_x = jnp.linspace(data.x.min(), data.x.max(), num=1000, endpoint=True)
        gp_prediction = Gaussian(
            **gp.predict_probability(
                parameters=gp_parameters,
                x=prediction_x,
            ).dict()
        )
        plotters.plot_data(
            train_data=data,
            inducing_data=Data(
                x=gp.x,
                y=gp.y,
            ),
            prediction_x=prediction_x,
            mean=gp_prediction.mean,
            covariance=gp_prediction.covariance,
            title=f"Reference GP Iteration {i}",
            save_path=os.path.join(
                checkpoint_path,
                f"iteration-{i}",
                f"reference-gp.png",
            ),
        )
    return gp, gp_parameters, post_epoch_histories


def train_approximate_gp(
    data: Data,
    empirical_risk_scheme: schemes.EmpiricalRisk,
    regularisation_scheme: schemes.Regularisation,
    trainer_settings: TrainerSettings,
    approximate_gp: ApproximateGPRegression,
    approximate_gp_parameters: ApproximateGPRegressionParameters,
    regulariser: GPRegressionBase,
    regulariser_parameters: GPBaseParameters,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
) -> Tuple[GPBaseParameters, List[Dict[str, float]]]:
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
            "empirical-risk": empirical_risk.calculate_empirical_risk(
                parameters, data.x, data.y
            ),
            "regularisation": regularisation.calculate_regularisation(
                parameters, data.x
            ),
            "gvi-objective": gvi.calculate_loss(parameters, data.x, data.y),
        },
    )
    gp_parameters, post_epoch_history = trainer.train(
        trainer_settings=trainer_settings,
        parameters=approximate_gp_parameters,
        data=data,
        loss_function=lambda parameters_dict, x, y: gvi.calculate_loss(
            parameters=parameters_dict, x=x, y=y
        ),
    )
    return approximate_gp.generate_parameters(gp_parameters.dict()), post_epoch_history


def train_tempered_gp(
    data: Data,
    empirical_risk_scheme: schemes.EmpiricalRisk,
    trainer_settings: TrainerSettings,
    tempered_gp: GPRegressionBase,
    tempered_gp_parameters: GPBaseParameters,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
) -> Tuple[GPBaseParameters, List[Dict[str, float]]]:
    empirical_risk = resolvers.empirical_risk(
        empirical_risk_scheme=empirical_risk_scheme,
        gp=tempered_gp,
    )
    trainer = Trainer(
        save_checkpoint_frequency=save_checkpoint_frequency,
        checkpoint_path=checkpoint_path,
        post_epoch_callback=lambda parameters: {
            "empirical-risk": empirical_risk.calculate_empirical_risk(
                parameters=tempered_gp_parameters.construct(
                    log_observation_noise=tempered_gp_parameters.log_observation_noise,
                    mean=tempered_gp_parameters.mean,
                    kernel=tempered_gp_parameters.kernel.construct(**parameters.dict()),
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
        loss_function=lambda parameters_dict, x, y: empirical_risk.calculate_empirical_risk(
            parameters=tempered_gp.Parameters(
                log_observation_noise=tempered_gp_parameters.log_observation_noise,
                mean=tempered_gp_parameters.mean,
                kernel=tempered_gp_parameters.kernel.construct(**parameters_dict),
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
