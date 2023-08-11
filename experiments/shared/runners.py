from typing import Dict, List, Tuple

from experiments.shared.data import Data
from experiments.shared.resolvers import (
    empirical_risk_resolver,
    regularisation_resolver,
)
from experiments.shared.schemes import EmpiricalRiskScheme, RegularisationScheme
from experiments.shared.trainer import Trainer, TrainerSettings
from src import GeneralisedVariationalInference
from src.gps.base.approximate_base import ApproximateGPBase
from src.gps.base.base import GPBase, GPBaseParameters


def train_approximate_gp(
    data: Data,
    empirical_risk_scheme: EmpiricalRiskScheme,
    regularisation_scheme: RegularisationScheme,
    trainer_settings: TrainerSettings,
    approximate_gp: ApproximateGPBase,
    approximate_gp_parameters: GPBaseParameters,
    regulariser: GPBase,
    regulariser_parameters: GPBaseParameters,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
) -> Tuple[GPBaseParameters, List[Dict[str, float]]]:
    empirical_risk = empirical_risk_resolver(
        empirical_risk_scheme=empirical_risk_scheme,
        gp=approximate_gp,
    )
    regularisation = regularisation_resolver(
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
    empirical_risk_scheme: EmpiricalRiskScheme,
    trainer_settings: TrainerSettings,
    tempered_gp: GPBase,
    tempered_gp_parameters: GPBaseParameters,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
) -> Tuple[GPBaseParameters, List[Dict[str, float]]]:
    empirical_risk = empirical_risk_resolver(
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
