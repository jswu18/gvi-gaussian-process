import os
from typing import Dict, Tuple, Union

import jax
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from experiments.regression.data import split_train_test_validation_data
from experiments.regression.plotters import plot_data
from experiments.regression.trainers import meta_train_reference_gp
from experiments.shared.data import Data, ExperimentData
from experiments.shared.resolvers import kernel_resolver, trainer_settings_resolver
from experiments.shared.schemas import (
    EmpiricalRiskSchema,
    KernelSchema,
    RegularisationSchema,
)
from experiments.shared.trainer import TrainerSettings
from src.utils.custom_types import PRNGKey


def run_set_up_experiment_data_chunked_test_data(
    key: PRNGKey,
    name: str,
    x: jnp.ndarray,
    y: jnp.ndarray,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    train_data_percentage: float,
    save_path: str,
) -> None:
    key, subkey = jax.random.split(key)
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
    experiment_data = ExperimentData(
        name=name,
        full=Data(x=x, y=y),
        train=Data(x=x_train, y=y_train),
        test=Data(x=x_test, y=y_test),
        validation=Data(x=x_validation, y=y_validation),
    )
    experiment_data.save(save_path)


def run_plot_experiment_data(
    experiment_data_path: str,
    name: str,
    title: str,
    save_path: str,
) -> None:
    experiment_data = ExperimentData.load(
        path=experiment_data_path,
        name=name,
    )
    plot_data(
        train_data=experiment_data.train,
        test_data=experiment_data.test,
        validation_data=experiment_data.validation,
        title=title,
        save_path=os.path.join(
            save_path,
            experiment_data.name,
            "data.png",
        ),
    )


def run_train_reference_model(
    experiment_data_path: str,
    name: str,
    kernel_config: Union[FrozenDict, Dict],
    empirical_risk_schema: EmpiricalRiskSchema,
    trainer_settings_config: Union[FrozenDict, Dict],
    number_of_inducing_points: int,
    number_of_iterations: int,
    empirical_risk_break_condition: float,
    save_checkpoint_frequency: int,
    save_path: str,
) -> None:
    experiment_data = ExperimentData.load(
        path=experiment_data_path,
        name=name,
    )
    trainer_settings = trainer_settings_resolver(
        trainer_settings_config=trainer_settings_config,
    )
    kernel, kernel_parameters = kernel_resolver(
        kernel_config=kernel_config,
    )
    (
        reference_gp,
        reference_gp_parameters,
        reference_post_epoch_histories,
    ) = meta_train_reference_gp(
        data=experiment_data.train,
        empirical_risk_schema=empirical_risk_schema,
        trainer_settings=trainer_settings,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
        number_of_inducing_points=number_of_inducing_points,
        number_of_iterations=number_of_iterations,
        empirical_risk_break_condition=empirical_risk_break_condition,
        save_checkpoint_frequency=save_checkpoint_frequency,
        checkpoint_path=os.path.join(
            save_path,
            experiment_data.name,
            "checkpoints",
        ),
    )
    reference_gp_parameters.save(
        path=os.path.join(
            save_path,
            experiment_data.name,
            "parameters.ckpt",
        ),
    )
