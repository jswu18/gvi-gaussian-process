import os
from typing import Dict, Tuple

import jax
import pandas as pd
import yaml
from jax import numpy as jnp

from experiments.regression.dataset_constants import DATASET_SCHEMA_TO_DATASET
from experiments.regression.metrics import calculate_metrics
from experiments.regression.trainers import meta_train_reference_gp
from experiments.shared.data import Data, ExperimentData, set_up_experiment
from experiments.shared.plotters import plot_losses, plot_two_losses
from experiments.shared.resolvers import (
    inducing_points_selector_resolver,
    kernel_resolver,
    mean_resolver,
    trainer_settings_resolver,
)
from experiments.shared.resolvers.kernel import resolve_existing_kernel
from experiments.shared.schemas import ActionSchema, EmpiricalRiskSchema
from experiments.shared.trainers import train_approximate_gp, train_tempered_gp
from experiments.shared.utils import construct_path
from src.gps import (
    ApproximateGPRegression,
    ApproximateGPRegressionParameters,
    GPRegression,
)
from src.kernels import TemperedKernel
from src.kernels.base import KernelBase, KernelBaseParameters
from src.means import ConstantMean


def _load_config_and_config_path(
    config_directory_parent_path: str, action: ActionSchema, name: str
) -> Tuple[Dict, str]:
    config_path = os.path.join(
        config_directory_parent_path,
        "configs",
        action,
        f"{name}.yaml",
    )
    with open(config_path, "r") as reference_config_file:
        config = yaml.safe_load(reference_config_file)
    return config, config_path


def build_data_set(
    config: Dict,
    output_path: str,
    experiment_name: str,
    rescale_y: bool,
) -> None:
    assert "seed" in config, "Seed must be specified for data set generation"
    assert "dataset" in config, "Dataset must be specified for data set generation"
    assert (
        "train_data_percentage" in config
    ), "Train data percentage must be specified for data set generation"
    assert (
        "test_data_percentage" in config
    ), "Test data percentage must be specified for data set generation"
    assert (
        "validation_data_percentage" in config
    ), "Validation data percentage must be specified for data set generation"

    data_csv_path = f"experiments/regression/datasets/{config['dataset']}.csv"
    df = pd.read_csv(data_csv_path)
    df.columns = [c.lower() for c in df.columns]
    df.columns = [c.replace(" ", "") for c in df.columns]

    dataset_metadata = DATASET_SCHEMA_TO_DATASET[config["dataset"]]
    input_column_names = [c.lower() for c in dataset_metadata.input_column_names]
    input_column_names = [c.replace(" ", "") for c in input_column_names]

    x = jnp.array(df[input_column_names].to_numpy())
    y = jnp.array(
        df[dataset_metadata.output_column_name.lower().replace(" ", "")].to_numpy()
    )

    data_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=ActionSchema.build_data,
    )
    experiment_data = set_up_experiment(
        name=config["dataset"],
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
    config: Dict,
    output_path: str,
    experiment_name: str,
    config_directory_parent_path: str,
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
    build_data_config, _ = _load_config_and_config_path(
        config_directory_parent_path=config_directory_parent_path,
        action=ActionSchema.build_data,
        name=config["data_name"],
    )
    save_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=ActionSchema.train_reference,
    )
    data_path = construct_path(
        output_path=output_path,
        experiment_name=config["data_name"],
        action=ActionSchema.build_data,
    )
    trainer_settings = trainer_settings_resolver(
        trainer_settings_config=config["trainer_settings"],
    )
    experiment_data = ExperimentData.load(
        path=data_path,
        name=build_data_config["dataset"],
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
        kernel_config=config["kernel"], data_dimension=experiment_data.train.x.shape[1]
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
    df_metrics = calculate_metrics(
        experiment_data=experiment_data,
        gp=reference_gp,
        gp_parameters=reference_gp_parameters,
        action=ActionSchema.train_reference,
        experiment_name=experiment_name,
        dataset_name=build_data_config["dataset"],
    )
    if not os.path.exists(os.path.join(save_path, experiment_data.name)):
        os.makedirs(os.path.join(save_path, experiment_data.name))
    df_metrics.to_csv(
        os.path.join(
            save_path,
            experiment_data.name,
            "metrics.csv",
        )
    )
    plot_losses(
        losses=[
            [x["empirical-risk"] for x in reference_post_epoch_history]
            for reference_post_epoch_history in reference_post_epoch_histories
        ],
        labels=[f"iteration-{i}" for i in range(len(reference_post_epoch_histories))],
        loss_name=config["empirical_risk_schema"],
        title=f"Reference GP Empirical Risk: {build_data_config['dataset']}",
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
        action=ActionSchema.train_reference,
    )
    save_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=ActionSchema.train_approximate,
    )
    trainer_settings = trainer_settings_resolver(
        trainer_settings_config=config["trainer_settings"],
    )
    reference_config, reference_config_path = _load_config_and_config_path(
        config_directory_parent_path=config_directory_parent_path,
        action=ActionSchema.train_reference,
        name=config["reference_name"],
    )
    build_data_config, _ = _load_config_and_config_path(
        config_directory_parent_path=config_directory_parent_path,
        action=ActionSchema.build_data,
        name=reference_config["data_name"],
    )
    data_path = construct_path(
        output_path=output_path,
        experiment_name=reference_config["data_name"],
        action=ActionSchema.build_data,
    )
    experiment_data = ExperimentData.load(
        path=data_path,
        name=build_data_config["dataset"],
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
                action=ActionSchema.train_reference,
            ),
            experiment_data.name,
            "parameters.ckpt",
        ),
        data_dimension=experiment_data.train.x.shape[1],
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
    approximate_gp_parameters, approximate_post_epoch_history = train_approximate_gp(
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
    df_metrics = calculate_metrics(
        experiment_data=experiment_data,
        gp=approximate_gp,
        gp_parameters=approximate_gp_parameters,
        action=ActionSchema.train_approximate,
        experiment_name=experiment_name,
        dataset_name=build_data_config["dataset"],
    )
    if not os.path.exists(os.path.join(save_path, experiment_data.name)):
        os.makedirs(os.path.join(save_path, experiment_data.name))
    df_metrics.to_csv(
        os.path.join(
            save_path,
            experiment_data.name,
            "metrics.csv",
        )
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
                f"{build_data_config['dataset']}",
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
                f"{build_data_config['dataset']}",
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
        data_dimension=training_points.shape[1],
    )
    kernel, kernel_parameters = kernel_resolver(
        kernel_config=config["kernel"],
        reference_kernel=reference_kernel,
        reference_kernel_parameters=reference_kernel_parameters,
        data_dimension=training_points.shape[1],
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
        action=ActionSchema.temper_approximate,
    )
    trainer_settings = trainer_settings_resolver(
        trainer_settings_config=config["trainer_settings"],
    )
    approximate_path = construct_path(
        output_path=output_path,
        experiment_name=config["approximate_name"],
        action=ActionSchema.train_approximate,
    )
    approximate_config, approximate_config_path = _load_config_and_config_path(
        config_directory_parent_path=config_directory_parent_path,
        action=ActionSchema.train_approximate,
        name=config["approximate_name"],
    )
    reference_config, reference_config_path = _load_config_and_config_path(
        config_directory_parent_path=config_directory_parent_path,
        action=ActionSchema.train_reference,
        name=approximate_config["reference_name"],
    )
    build_data_config, _ = _load_config_and_config_path(
        config_directory_parent_path=config_directory_parent_path,
        action=ActionSchema.build_data,
        name=reference_config["data_name"],
    )
    data_path = construct_path(
        output_path=output_path,
        experiment_name=reference_config["data_name"],
        action=ActionSchema.build_data,
    )
    experiment_data = ExperimentData.load(
        path=data_path,
        name=build_data_config["dataset"],
    )
    reference_path = construct_path(
        output_path=output_path,
        experiment_name=approximate_config["reference_name"],
        action=ActionSchema.train_reference,
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
                action=ActionSchema.train_reference,
            ),
            experiment_data.name,
            "parameters.ckpt",
        ),
        data_dimension=experiment_data.train.x.shape[1],
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
    df_metrics = calculate_metrics(
        experiment_data=experiment_data,
        gp=tempered_gp,
        gp_parameters=tempered_gp_parameters,
        action=ActionSchema.temper_approximate,
        experiment_name=experiment_name,
        dataset_name=build_data_config["dataset"],
    )
    if not os.path.exists(os.path.join(save_path, experiment_data.name)):
        os.makedirs(os.path.join(save_path, experiment_data.name))
    df_metrics.to_csv(
        os.path.join(
            save_path,
            experiment_data.name,
            "metrics.csv",
        )
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
                f"{build_data_config['dataset']}",
            ]
        ),
        save_path=os.path.join(
            save_path,
            experiment_data.name,
            "loss.png",
        ),
    )
