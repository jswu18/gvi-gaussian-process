import operator
import os
from functools import reduce
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from tqdm import tqdm

from experiments.classification.plotters import plot_images
from experiments.shared.data import Data
from experiments.shared.resolvers import empirical_risk_resolver
from experiments.shared.schemes import EmpiricalRiskScheme
from experiments.shared.trainer import Trainer, TrainerSettings
from experiments.shared.utils import calculate_inducing_points
from src.gps import GPClassification, GPClassificationParameters
from src.kernels import MultiOutputKernel, MultiOutputKernelParameters
from src.means import ConstantMean


def train_reference_gp(
    data_list: List[Data],
    empirical_risk_scheme: EmpiricalRiskScheme,
    trainer_settings: TrainerSettings,
    kernel: MultiOutputKernel,
    kernel_parameters: MultiOutputKernelParameters,
    number_of_inducing_per_label: int,
    empirical_risk_break_condition: float,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
) -> Tuple[
    GPClassification, GPClassificationParameters, List[Dict[str, float]], List[Data]
]:
    assert len(data_list) == len(kernel.kernels), (
        f"Number of data sets ({len(data_list)}) must match number of kernels "
        f"({len(kernel.kernels)})"
    )
    assert len(data_list) == len(kernel_parameters.kernels), (
        f"Number of data sets ({len(data_list)}) must match number of kernel "
        f"parameters ({len(kernel_parameters.kernels)})"
    )
    inducing_data_list: List[Data] = []
    # use each kernel in the multi-output kernel to calculate inducing points for each class
    for data, single_label_kernel, single_label_kernel_parameters in zip(
        data_list, kernel.kernels, kernel_parameters.kernels
    ):
        inducing_data_list.append(
            calculate_inducing_points(
                key=jax.random.PRNGKey(trainer_settings.key),
                data=data,
                number_of_inducing_points=number_of_inducing_per_label,
                kernel=single_label_kernel,
                kernel_parameters=single_label_kernel_parameters,
            )
        )
    inducing_data_combined = reduce(operator.add, inducing_data_list)

    gp = GPClassification(
        x=inducing_data_combined.x,
        y=inducing_data_combined.y,
        kernel=kernel,
        mean=ConstantMean(number_output_dimensions=len(data_list)),
    )
    gp_parameters = gp.generate_parameters(
        {
            "log_observation_noise": jnp.log(jnp.ones((len(data_list),))),
            "mean": {"constant": jnp.zeros((len(data_list),))},
            "kernel": kernel_parameters.dict(),
        }
    )
    empirical_risk = empirical_risk_resolver(
        empirical_risk_scheme=empirical_risk_scheme,
        gp=gp,
    )
    trainer = Trainer(
        save_checkpoint_frequency=save_checkpoint_frequency,
        checkpoint_path=checkpoint_path,
        post_epoch_callback=lambda parameters: {
            "empirical-risk": empirical_risk.calculate_empirical_risk(
                parameters, inducing_data_combined.x, inducing_data_combined.y
            )
        },
        break_condition_function=(
            lambda parameters: empirical_risk.calculate_empirical_risk(
                parameters, inducing_data_combined.x, inducing_data_combined.y
            )
            < empirical_risk_break_condition
        ),
    )
    gp_parameters, post_epoch_history = trainer.train(
        trainer_settings=trainer_settings,
        parameters=gp_parameters,
        data=inducing_data_combined,
        loss_function=lambda parameters_dict, x, y: empirical_risk.calculate_empirical_risk(
            parameters=parameters_dict,
            x=x,
            y=y,
        ),
        disable_tqdm=True,
    )
    gp_parameters = gp.generate_parameters(gp_parameters.dict())
    return gp, gp_parameters, post_epoch_history, inducing_data_list


def meta_train_reference_gp(
    data_list: List[Data],
    empirical_risk_scheme: EmpiricalRiskScheme,
    trainer_settings: TrainerSettings,
    kernel: MultiOutputKernel,
    kernel_parameters: MultiOutputKernelParameters,
    number_of_inducing_per_label: int,
    number_of_iterations: int,
    save_checkpoint_frequency: int,
    checkpoint_path: str,
    empirical_risk_break_condition: float = -float("inf"),
) -> Tuple[
    GPClassification,
    GPClassificationParameters,
    List[List[Dict[str, float]]],
    List[Data],
]:
    post_epoch_histories: List[List[Dict[str, float]]] = []
    gp, gp_parameters = None, None
    inducing_data_list: List[Data] = []
    for i in tqdm(range(number_of_iterations)):
        gp, gp_parameters, post_epoch_history, inducing_data_list = train_reference_gp(
            data_list=data_list,
            empirical_risk_scheme=empirical_risk_scheme,
            trainer_settings=trainer_settings,
            kernel=kernel,
            kernel_parameters=kernel_parameters,
            number_of_inducing_per_label=number_of_inducing_per_label,
            empirical_risk_break_condition=empirical_risk_break_condition,
            save_checkpoint_frequency=save_checkpoint_frequency,
            checkpoint_path=os.path.join(checkpoint_path, f"iteration-{i}"),
        )
        kernel_parameters = gp_parameters.kernel
        post_epoch_histories.append(post_epoch_history)
        plot_images(
            data_list=inducing_data_list,
            reshape_function=lambda x: x.reshape(28, 28),
            save_path=os.path.join(
                checkpoint_path, f"inducing-images-iteration-{i}.png"
            ),
        )
    return gp, gp_parameters, post_epoch_histories, inducing_data_list
