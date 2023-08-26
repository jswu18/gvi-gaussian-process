import os
from typing import Dict, List, Tuple

import jax
import pandas as pd
import yaml
from jax import numpy as jnp
from jax import scipy as jsp

from experiments.regression.trainers import meta_train_reference_gp
from experiments.shared.data import Data, ExperimentData, set_up_experiment
from experiments.shared.plotters import plot_losses, plot_two_losses
from experiments.shared.resolvers import (
    empirical_risk_resolver,
    inducing_points_selector_resolver,
    kernel_resolver,
    mean_resolver,
    trainer_settings_resolver,
)
from experiments.shared.resolvers.kernel import resolve_existing_kernel
from experiments.shared.schemas import Actions, EmpiricalRiskSchema
from experiments.shared.trainers import train_approximate_gp, train_tempered_gp
from experiments.shared.utils import construct_path
from src.distributions import Gaussian
from src.gps import (
    ApproximateGPRegression,
    ApproximateGPRegressionParameters,
    GPRegression,
)
from src.kernels import TemperedKernel
from src.kernels.base import KernelBase, KernelBaseParameters
from src.means import ConstantMean


def build_data_set(
    config: Dict,
    data_csv_path: str,
    output_path: str,
    experiment_name: str,
    dataset_name: str,
    rescale_y: bool,
) -> None:
    assert "seed" in config, "Seed must be specified for data set generation"
    assert (
        "train_data_percentage" in config
    ), "Train data percentage must be specified for data set generation"
    assert (
        "test_data_percentage" in config
    ), "Test data percentage must be specified for data set generation"
    assert (
        "validation_data_percentage" in config
    ), "Validation data percentage must be specified for data set generation"

    df = pd.read_csv(data_csv_path)

    # assume last column is regression target
    x = jnp.array(df[df.columns[:-1]].to_numpy())
    y = jnp.array(df[df.columns[-1]].to_numpy())

    data_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=Actions.build_data,
    )
    experiment_data = set_up_experiment(
        name=dataset_name,
        key=jax.random.PRNGKey(config["seed"]),
        x=x,
        y=y,
        train_data_percentage=config["train_data_percentage"],
        test_data_percentage=config["test_data_percentage"],
        validation_data_percentage=config["validation_data_percentage"],
        rescale_y=rescale_y,
    )
    experiment_data.save(data_path)


def train_reference(
    config: Dict, output_path: str, experiment_name: str, dataset_name: str
) -> None:
    assert "data_name" in config, "Data name must be specified for reference training"
    assert (
        "trainer_settings" in config
    ), "Trainer settings must be specified for reference training"
    assert "kernel" in config, "Kernel config must be specified for reference training"
    assert (
        "inducing_points" in config
    ), "Inducing points must be specified for reference training"
    assert (
        "inducing_points_factor" in config["inducing_points"]
    ), "Inducing points factor must be specified for reference training"
    assert (
        "inducing_points_power" in config["inducing_points"]
    ), "Inducing points power must be specified for reference training"
    assert (
        "inducing_points_selector_schema" in config["inducing_points"]
    ), "Inducing points schema must be specified for reference training"
    assert (
        "number_of_iterations" in config
    ), "Number of iterations must be specified for reference training"
    assert (
        "empirical_risk_schema" in config
    ), "Empirical risk schema must be specified for reference training"
    assert (
        "empirical_risk_break_condition" in config
    ), "Empirical risk break condition must be specified for reference training"
    assert (
        "save_checkpoint_frequency" in config
    ), "Save checkpoint frequency must be specified for reference training"
    assert "kernel" in config, "Kernel must be specified for reference training"
    save_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=Actions.train_reference,
    )
    data_path = construct_path(
        output_path=output_path,
        experiment_name=config["data_name"],
        action=Actions.build_data,
    )
    trainer_settings = trainer_settings_resolver(
        trainer_settings_config=config["trainer_settings"],
    )
    experiment_data = ExperimentData.load(
        path=data_path,
        name=dataset_name,
    )
    number_of_inducing_points = int(
        config["inducing_points"]["inducing_points_factor"]
        * jnp.power(
            len(experiment_data.train.x),
            1 / config["inducing_points"]["inducing_points_power"],
        )
    )
    inducing_points_selector = inducing_points_selector_resolver(
        inducing_points_schema=config["inducing_points"][
            "inducing_points_selector_schema"
        ],
    )
    kernel, kernel_parameters = kernel_resolver(
        kernel_config=config["kernel"],
    )
    (
        reference_gp,
        reference_gp_parameters,
        reference_post_epoch_histories,
    ) = meta_train_reference_gp(
        data=experiment_data.train,
        empirical_risk_schema=config["empirical_risk_schema"],
        trainer_settings=trainer_settings,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
        inducing_points_selector=inducing_points_selector,
        number_of_inducing_points=number_of_inducing_points,
        number_of_iterations=config["number_of_iterations"],
        empirical_risk_break_condition=config["empirical_risk_break_condition"],
        save_checkpoint_frequency=config["save_checkpoint_frequency"],
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
    inducing_data = Data(x=reference_gp.x, y=reference_gp.y, name="inducing")
    inducing_data.save(
        path=os.path.join(
            save_path,
            experiment_data.name,
        ),
    )
    plot_losses(
        losses=[
            [x["empirical-risk"] for x in reference_post_epoch_history]
            for reference_post_epoch_history in reference_post_epoch_histories
        ],
        labels=[f"iteration-{i}" for i in range(len(reference_post_epoch_histories))],
        loss_name=config["empirical_risk_schema"],
        title=f"Reference GP Empirical Risk: {dataset_name}",
        save_path=os.path.join(
            save_path,
            experiment_data.name,
            "losses.png",
        ),
    )


def train_approximate(
    config: Dict,
    output_path: str,
    experiment_name: str,
    dataset_name: str,
    config_directory_parent_path: str,
) -> None:
    assert (
        "reference_name" in config
    ), "Reference name must be specified for approximate training"
    assert (
        "trainer_settings" in config
    ), "Trainer settings must be specified for approximate training"
    assert (
        "kernel" in config
    ), "Kernel config must be specified for approximate training"
    assert (
        "empirical_risk_schema" in config
    ), "Empirical risk schema must be specified for approximate training"
    assert (
        "regularisation" in config
    ), "Regularisation must be specified for approximate training"
    assert (
        "save_checkpoint_frequency" in config
    ), "Save checkpoint frequency must be specified for approximate training"
    assert "mean" in config, "Mean must be specified for reference training"
    assert "kernel" in config, "Kernel must be specified for reference training"
    reference_path = construct_path(
        output_path=output_path,
        experiment_name=config["reference_name"],
        action=Actions.train_reference,
    )
    save_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=Actions.train_approximate,
    )
    trainer_settings = trainer_settings_resolver(
        trainer_settings_config=config["trainer_settings"],
    )
    reference_config_path = os.path.join(
        config_directory_parent_path,
        "configs",
        Actions.train_reference,
        f"{config['reference_name']}.yaml",
    )
    with open(reference_config_path, "r") as reference_config_file:
        reference_config = yaml.safe_load(reference_config_file)
    data_path = construct_path(
        output_path=output_path,
        experiment_name=reference_config["data_name"],
        action=Actions.build_data,
    )
    experiment_data = ExperimentData.load(
        path=data_path,
        name=dataset_name,
    )
    inducing_data = Data.load(
        path=os.path.join(
            reference_path,
            experiment_data.name,
        ),
        name="inducing",
    )
    reference_kernel, reference_kernel_parameters = resolve_existing_kernel(
        config_path=reference_config_path,
        parameter_path=os.path.join(
            construct_path(
                output_path=output_path,
                experiment_name=config["reference_name"],
                action=Actions.train_reference,
            ),
            experiment_data.name,
            "parameters.ckpt",
        ),
    )
    approximate_gp, initial_approximate_gp_parameters = build_approximate_gp(
        config=config,
        inducing_points=inducing_data.x,
        training_points=experiment_data.train.x,
        reference_kernel=reference_kernel,
        reference_kernel_parameters=reference_kernel_parameters,
    )
    regulariser_kernel = reference_kernel
    regulariser = GPRegression(
        x=inducing_data.x,
        y=inducing_data.y,
        kernel=regulariser_kernel,
        mean=ConstantMean(),
    )
    regulariser_parameters = regulariser.generate_parameters(
        regulariser.Parameters.load(
            regulariser.Parameters,
            path=os.path.join(
                reference_path,
                experiment_data.name,
                "parameters.ckpt",
            ),
        ).dict()
    )
    (approximate_gp_parameters, approximate_post_epoch_history,) = train_approximate_gp(
        data=experiment_data.train,
        empirical_risk_schema=config["empirical_risk_schema"],
        regularisation_config=config["regularisation"],
        trainer_settings=trainer_settings,
        approximate_gp=approximate_gp,
        approximate_gp_parameters=initial_approximate_gp_parameters,
        regulariser=regulariser,
        regulariser_parameters=regulariser_parameters,
        save_checkpoint_frequency=config["save_checkpoint_frequency"],
        checkpoint_path=os.path.join(
            save_path,
            experiment_data.name,
            "checkpoints",
        ),
    )
    approximate_gp_parameters.save(
        path=os.path.join(
            save_path,
            experiment_data.name,
            "parameters.ckpt",
        ),
    )
    gvi_loss_configuration = "+".join(
        [
            config["empirical_risk_schema"],
            config["regularisation"]["regularisation_schema"],
        ]
    )
    plot_losses(
        losses=[x["gvi-objective"] for x in approximate_post_epoch_history],
        labels="gvi-objective",
        loss_name=f"{config['empirical_risk_schema']}+{config['regularisation']['regularisation_schema']}",
        title=" ".join(
            [
                f"Approximate GP Objective ({gvi_loss_configuration}):",
                f"{dataset_name}",
            ]
        ),
        save_path=os.path.join(
            save_path,
            experiment_data.name,
            "loss.png",
        ),
    )
    plot_two_losses(
        loss1=[x["empirical-risk"] for x in approximate_post_epoch_history],
        loss1_name=config["empirical_risk_schema"],
        loss2=[x["regularisation"] for x in approximate_post_epoch_history],
        loss2_name=config["regularisation"]["regularisation_schema"],
        title=" ".join(
            [
                f"Approximate GP Objective Breakdown ({gvi_loss_configuration}):",
                f"{dataset_name}",
            ]
        ),
        save_path=os.path.join(
            save_path,
            experiment_data.name,
            "loss-breakdown.png",
        ),
    )


def build_approximate_gp(
    config: Dict,
    inducing_points: jnp.ndarray,
    training_points: jnp.ndarray,
    reference_kernel: KernelBase,
    reference_kernel_parameters: KernelBaseParameters,
) -> Tuple[ApproximateGPRegression, ApproximateGPRegressionParameters]:
    config["kernel"]["kernel_kwargs"]["inducing_points"] = inducing_points
    config["kernel"]["kernel_kwargs"]["training_points"] = training_points

    mean, mean_parameters = mean_resolver(
        mean_config=config["mean"],
    )
    kernel, kernel_parameters = kernel_resolver(
        kernel_config=config["kernel"],
        reference_kernel=reference_kernel,
        reference_kernel_parameters=reference_kernel_parameters,
    )
    gp = ApproximateGPRegression(
        mean=mean,
        kernel=kernel,
    )
    gp_parameters = gp.generate_parameters(
        {
            "mean": mean_parameters,
            "kernel": kernel_parameters,
        }
    )
    return gp, gp_parameters


def temper_approximate(
    config: Dict,
    output_path: str,
    experiment_name: str,
    dataset_name: str,
    config_directory_parent_path: str,
) -> None:
    assert "approximate_name" in config, "Approximate name must be specified"
    assert (
        "trainer_settings" in config
    ), "Trainer settings must be specified for approximate training"
    assert (
        "save_checkpoint_frequency" in config
    ), "Save checkpoint frequency must be specified for approximate training"
    save_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=Actions.temper_approximate,
    )
    trainer_settings = trainer_settings_resolver(
        trainer_settings_config=config["trainer_settings"],
    )
    approximate_path = construct_path(
        output_path=output_path,
        experiment_name=config["approximate_name"],
        action=Actions.train_approximate,
    )
    approximate_config_path = os.path.join(
        config_directory_parent_path,
        "configs",
        Actions.train_approximate,
        f"{config['approximate_name']}.yaml",
    )
    with open(approximate_config_path, "r") as approximate_config_file:
        approximate_config = yaml.safe_load(approximate_config_file)
    reference_config_path = os.path.join(
        config_directory_parent_path,
        "configs",
        Actions.train_reference,
        f"{approximate_config['reference_name']}.yaml",
    )
    with open(reference_config_path, "r") as reference_config_file:
        reference_config = yaml.safe_load(reference_config_file)
    data_path = construct_path(
        output_path=output_path,
        experiment_name=reference_config["data_name"],
        action=Actions.build_data,
    )
    experiment_data = ExperimentData.load(
        path=data_path,
        name=dataset_name,
    )
    approximate_config_path = os.path.join(
        os.path.join(
            config_directory_parent_path,
            "configs",
            Actions.train_approximate,
            f"{config['approximate_name']}.yaml",
        )
    )

    with open(approximate_config_path, "r") as approximate_config_file:
        approximate_config = yaml.safe_load(approximate_config_file)
    reference_path = construct_path(
        output_path=output_path,
        experiment_name=approximate_config["reference_name"],
        action=Actions.train_reference,
    )
    inducing_data = Data.load(
        path=os.path.join(
            reference_path,
            experiment_data.name,
        ),
        name="inducing",
    )
    reference_kernel, reference_kernel_parameters = resolve_existing_kernel(
        config_path=reference_config_path,
        parameter_path=os.path.join(
            construct_path(
                output_path=output_path,
                experiment_name=approximate_config["reference_name"],
                action=Actions.train_reference,
            ),
            experiment_data.name,
            "parameters.ckpt",
        ),
    )
    approximate_gp, _ = build_approximate_gp(
        config=approximate_config,
        inducing_points=inducing_data.x,
        training_points=experiment_data.train.x,
        reference_kernel=reference_kernel,
        reference_kernel_parameters=reference_kernel_parameters,
    )
    approximate_gp_parameters = approximate_gp.generate_parameters(
        approximate_gp.Parameters.load(
            approximate_gp.Parameters,
            path=os.path.join(
                approximate_path,
                experiment_data.name,
                "parameters.ckpt",
            ),
        ).dict()
    )
    tempered_gp = type(approximate_gp)(
        mean=approximate_gp.mean,
        kernel=TemperedKernel(
            base_kernel=approximate_gp.kernel,
            base_kernel_parameters=approximate_gp_parameters.kernel,
            number_output_dimensions=approximate_gp.kernel.number_output_dimensions,
        ),
    )
    initial_tempered_gp_parameters = approximate_gp.Parameters.construct(
        log_observation_noise=approximate_gp_parameters.log_observation_noise,
        mean=approximate_gp_parameters.mean,
        kernel=TemperedKernel.Parameters.construct(log_tempering_factor=jnp.log(1.0)),
    )
    tempered_gp_parameters, tempered_post_epoch_history = train_tempered_gp(
        data=experiment_data.validation,
        empirical_risk_schema=EmpiricalRiskSchema.negative_log_likelihood,
        trainer_settings=trainer_settings,
        tempered_gp=tempered_gp,
        tempered_gp_parameters=initial_tempered_gp_parameters,
        save_checkpoint_frequency=config["save_checkpoint_frequency"],
        checkpoint_path=os.path.join(
            save_path,
            experiment_data.name,
            "checkpoints",
        ),
    )
    approximate_gp_gvi_loss_configuration = "+".join(
        [
            approximate_config["empirical_risk_schema"],
            approximate_config["regularisation"]["regularisation_schema"],
        ]
    )
    plot_losses(
        losses=[x["empirical-risk"] for x in tempered_post_epoch_history],
        labels="empirical-risk",
        loss_name=EmpiricalRiskSchema.negative_log_likelihood,
        title=" ".join(
            [
                f"Tempered Approximate GP Empirical Risk ({approximate_gp_gvi_loss_configuration}):",
                f"{dataset_name}",
            ]
        ),
        save_path=os.path.join(
            save_path,
            experiment_data.name,
            "loss.png",
        ),
    )
    gaussian = Gaussian(
        **tempered_gp.predict_probability(
            tempered_gp_parameters, x=experiment_data.test.x
        ).dict()
    )
    out = []
    for loc, variance, y in zip(
        gaussian.mean, gaussian.covariance, experiment_data.test.y
    ):
        out.append(
            -jsp.stats.norm.logpdf(
                y,
                loc=float(loc),
                scale=float(jnp.sqrt(variance)),
            )
        )
    print(jnp.mean(jnp.array(out)) + jnp.log(experiment_data.y_std))
