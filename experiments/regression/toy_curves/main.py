import argparse
import enum
import os
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import yaml

from experiments.regression.data import split_train_test_validation_data
from experiments.regression.plotters import plot_data, plot_prediction
from experiments.regression.toy_curves.curves import CURVE_FUNCTIONS
from experiments.regression.trainers import meta_train_reference_gp
from experiments.shared.data import Data, ExperimentData
from experiments.shared.plotters import plot_losses, plot_two_losses
from experiments.shared.resolvers import (
    kernel_resolver,
    mean_resolver,
    trainer_settings_resolver,
)
from experiments.shared.schemas import EmpiricalRiskSchema
from experiments.shared.trainers import train_approximate_gp, train_tempered_gp
from src.gps import (
    ApproximateGPRegression,
    ApproximateGPRegressionParameters,
    GPRegression,
)
from src.kernels import TemperedKernel
from src.means import ConstantMean


class Actions(str, enum.Enum):
    build_data = "build_data"
    train_reference = "train_reference"
    train_approximate = "train_approximate"
    temper_approximate = "temper_approximate"


parser = argparse.ArgumentParser(description="Main script for toy curves experiments.")
parser.add_argument("--action", choices=[Actions[a].value for a in Actions])
parser.add_argument("--config_path", type=str)


def construct_path(output_path: str, experiment_name: str, action: Actions) -> str:
    return os.path.join(
        output_path,
        action,
        experiment_name,
    )


def build_data_set(config: Dict, output_path: str, experiment_name: str) -> None:
    assert "seed" in config, "Seed must be specified for data set generation"
    assert "number_of_data_points" in config, "Number of data points must be specified."
    assert (
        "number_of_test_intervals" in config
    ), "Number of test intervals must be specified."
    assert (
        "total_number_of_intervals" in config
    ), "Total number of intervals must be specified."
    assert "train_data_percentage" in config, "Train data percentage must be specified."
    assert "sigma_true" in config, "Sigma true must be specified."

    x = jnp.linspace(-2, 2, config["number_of_data_points"]).reshape(-1, 1)
    data_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=Actions.build_data,
    )
    for curve_function in CURVE_FUNCTIONS:
        y = curve_function(
            key=jax.random.PRNGKey(curve_function.seed),
            x=x,
            sigma_true=config["sigma_true"],
        )
        (
            x_train,
            y_train,
            x_test,
            y_test,
            x_validation,
            y_validation,
        ) = split_train_test_validation_data(
            key=jax.random.PRNGKey(config["seed"]),
            x=x,
            y=y,
            number_of_test_intervals=config["number_of_test_intervals"],
            total_number_of_intervals=config["total_number_of_intervals"],
            train_data_percentage=config["train_data_percentage"],
        )
        experiment_data = ExperimentData(
            name=type(curve_function).__name__.lower(),
            full=Data(x=x, y=y),
            train=Data(x=x_train, y=y_train),
            test=Data(x=x_test, y=y_test),
            validation=Data(x=x_validation, y=y_validation),
        )
        experiment_data.save(data_path)

        plot_data(
            train_data=experiment_data.train,
            test_data=experiment_data.test,
            validation_data=experiment_data.validation,
            title=f"{experiment_name}: {curve_function.__name__}",
            save_path=os.path.join(
                data_path,
                experiment_data.name,
                "data.png",
            ),
        )


def train_reference(config: Dict, output_path: str, experiment_name: str) -> None:
    assert "data_name" in config, "Data name must be specified for reference training"
    assert (
        "trainer_settings" in config
    ), "Trainer settings must be specified for reference training"
    assert "kernel" in config, "Kernel config must be specified for reference training"
    assert (
        "inducing_points_factor" in config
    ), "Inducing points factor must be specified for reference training"
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
    for curve_function in CURVE_FUNCTIONS:
        experiment_data = ExperimentData.load(
            path=data_path,
            name=type(curve_function).__name__.lower(),
        )
        number_of_inducing_points = int(
            config["inducing_points_factor"] * jnp.sqrt(len(experiment_data.train.x))
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
        plot_prediction(
            experiment_data=experiment_data,
            inducing_data=Data(
                x=reference_gp.x,
                y=reference_gp.y,
            ),
            gp=reference_gp,
            gp_parameters=reference_gp_parameters,
            title=f"Reference GP: {curve_function.__name__}",
            save_path=os.path.join(
                save_path,
                experiment_data.name,
                "prediction.png",
            ),
        )
        plot_losses(
            losses=[
                [x["empirical-risk"] for x in reference_post_epoch_history]
                for reference_post_epoch_history in reference_post_epoch_histories
            ],
            labels=[
                f"iteration-{i}" for i in range(len(reference_post_epoch_histories))
            ],
            loss_name=config["empirical_risk_schema"],
            title=f"Reference GP Empirical Risk: {curve_function.__name__}",
            save_path=os.path.join(
                save_path,
                experiment_data.name,
                "reference-gp-losses.png",
            ),
        )


def _build_approximate_gp(
    config: Dict,
    config_directory_parent_path: str,
    inducing_points: jnp.ndarray,
    training_points: jnp.ndarray,
    output_path: str,
    experiment_data_name: str,
) -> Tuple[ApproximateGPRegression, ApproximateGPRegressionParameters]:
    config["kernel"]["kernel_kwargs"]["inducing_points"] = inducing_points
    config["kernel"]["kernel_kwargs"]["training_points"] = training_points

    if "reference_kernel" in config["kernel"]["kernel_kwargs"]:
        reference_kernel_config_name = config["kernel"]["kernel_kwargs"][
            "reference_kernel"
        ]["load_config"]
        config["kernel"]["kernel_kwargs"]["reference_kernel"]["load_config_paths"] = {
            "config_path": os.path.join(
                config_directory_parent_path,
                "configs",
                Actions.train_reference,
                f"{reference_kernel_config_name}.yaml",
            ),
            "parameters_path": os.path.join(
                construct_path(
                    output_path=output_path,
                    experiment_name=reference_kernel_config_name,
                    action=Actions.train_reference,
                ),
                experiment_data_name,
                "parameters.ckpt",
            ),
        }
    if "base_kernel" in config["kernel"]["kernel_kwargs"]:
        base_kernel_config_name = config["kernel"]["kernel_kwargs"]["base_kernel"][
            "load_config"
        ]
        config["kernel"]["kernel_kwargs"]["base_kernel"]["load_config_paths"] = {
            "config_path": os.path.join(
                config_directory_parent_path,
                "configs",
                Actions.train_reference,
                f"{base_kernel_config_name}.yaml",
            ),
            "parameters_path": os.path.join(
                construct_path(
                    output_path=output_path,
                    experiment_name=base_kernel_config_name,
                    action=Actions.train_reference,
                ),
                experiment_data_name,
                "parameters.ckpt",
            ),
        }
    kernel, kernel_parameters = kernel_resolver(
        kernel_config=config["kernel"],
    )
    mean, mean_parameters = mean_resolver(
        mean_config=config["mean"],
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


def train_approximate(config: Dict, output_path: str, experiment_name: str) -> None:
    assert "data_name" in config, "Data name must be specified for approximate training"
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
        "regularisation_schema" in config
    ), "Regularisation schema must be specified for approximate training"
    assert (
        "save_checkpoint_frequency" in config
    ), "Save checkpoint frequency must be specified for approximate training"
    assert "mean" in config, "Mean must be specified for reference training"
    assert "kernel" in config, "Kernel must be specified for reference training"
    data_path = construct_path(
        output_path=output_path,
        experiment_name=config["data_name"],
        action=Actions.build_data,
    )
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
    for curve_function in CURVE_FUNCTIONS:
        experiment_data = ExperimentData.load(
            path=data_path,
            name=type(curve_function).__name__.lower(),
        )
        inducing_data = Data.load(
            path=os.path.join(
                reference_path,
                experiment_data.name,
            ),
            name="inducing",
        )
        approximate_gp, initial_approximate_gp_parameters = _build_approximate_gp(
            config=config,
            config_directory_parent_path=os.path.dirname(os.path.abspath(__file__)),
            inducing_points=inducing_data.x,
            training_points=experiment_data.train.x,
            output_path=output_path,
            experiment_data_name=experiment_data.name,
        )
        regulariser_kernel_config_name = config["reference_name"]
        regulariser_kernel, _ = kernel_resolver(
            kernel_config={
                "load_config_paths": {
                    "config_path": os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "configs",
                        Actions.train_reference,
                        f"{regulariser_kernel_config_name}.yaml",
                    ),
                    "parameters_path": os.path.join(
                        construct_path(
                            output_path=output_path,
                            experiment_name=regulariser_kernel_config_name,
                            action=Actions.train_reference,
                        ),
                        experiment_data.name,
                        "parameters.ckpt",
                    ),
                }
            },
        )
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
        (
            approximate_gp_parameters,
            approximate_post_epoch_history,
        ) = train_approximate_gp(
            data=experiment_data.train,
            empirical_risk_schema=config["empirical_risk_schema"],
            regularisation_schema=config["regularisation_schema"],
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
                config["regularisation_schema"],
            ]
        )
        plot_prediction(
            experiment_data=experiment_data,
            inducing_data=inducing_data,
            gp=approximate_gp,
            gp_parameters=approximate_gp_parameters,
            title=" ".join(
                [
                    f"Approximate GP ({gvi_loss_configuration}):",
                    f"{curve_function.__name__}",
                ]
            ),
            save_path=os.path.join(
                save_path,
                experiment_data.name,
                "approximate-gp.png",
            ),
        )
        plot_losses(
            losses=[x["gvi-objective"] for x in approximate_post_epoch_history],
            labels="gvi-objective",
            loss_name=f"{config['empirical_risk_schema']}+{config['regularisation_schema']}",
            title=" ".join(
                [
                    f"Approximate GP Objective ({gvi_loss_configuration}):",
                    f"{curve_function.__name__}",
                ]
            ),
            save_path=os.path.join(
                save_path,
                experiment_data.name,
                "approximate-gp-loss.png",
            ),
        )
        plot_two_losses(
            loss1=[x["empirical-risk"] for x in approximate_post_epoch_history],
            loss1_name=config["empirical_risk_schema"],
            loss2=[x["regularisation"] for x in approximate_post_epoch_history],
            loss2_name=config["regularisation_schema"],
            title=" ".join(
                [
                    f"Approximate GP Objective Breakdown ({gvi_loss_configuration}):",
                    f"{curve_function.__name__}",
                ]
            ),
            save_path=os.path.join(
                save_path,
                experiment_data.name,
                "approximate-gp-loss-breakdown.png",
            ),
        )


def temper_approximate(config: Dict, output_path: str, experiment_name: str) -> None:
    assert "data_name" in config, "Data name must be specified for approximate training"
    assert (
        "trainer_settings" in config
    ), "Trainer settings must be specified for approximate training"
    assert (
        "save_checkpoint_frequency" in config
    ), "Save checkpoint frequency must be specified for approximate training"
    data_path = construct_path(
        output_path=output_path,
        experiment_name=config["data_name"],
        action=Actions.build_data,
    )
    approximate_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=Actions.train_approximate,
    )
    save_path = construct_path(
        output_path=output_path,
        experiment_name=experiment_name,
        action=Actions.temper_approximate,
    )
    trainer_settings = trainer_settings_resolver(
        trainer_settings_config=config["trainer_settings"],
    )
    for curve_function in CURVE_FUNCTIONS:
        experiment_data = ExperimentData.load(
            path=data_path,
            name=type(curve_function).__name__.lower(),
        )
        approximate_config_path = os.path.join(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
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
        approximate_gp, _ = _build_approximate_gp(
            config=approximate_config,
            config_directory_parent_path=os.path.dirname(os.path.abspath(__file__)),
            inducing_points=inducing_data.x,
            training_points=experiment_data.train.x,
            output_path=output_path,
            experiment_data_name=experiment_data.name,
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
            kernel=TemperedKernel.Parameters.construct(
                log_tempering_factor=jnp.log(1.0)
            ),
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
                approximate_config["regularisation_schema"],
            ]
        )
        plot_prediction(
            experiment_data=experiment_data,
            inducing_data=inducing_data,
            gp=tempered_gp,
            gp_parameters=tempered_gp_parameters,
            title=" ".join(
                [
                    f"Tempered Approximate GP ({approximate_gp_gvi_loss_configuration}):",
                    f"{curve_function.__name__}",
                ]
            ),
            save_path=os.path.join(
                save_path, experiment_data.name, "tempered-approximate-gp.png"
            ),
        )
        plot_losses(
            losses=[x["empirical-risk"] for x in tempered_post_epoch_history],
            labels="empirical-risk",
            loss_name=EmpiricalRiskSchema.negative_log_likelihood,
            title=" ".join(
                [
                    f"Tempered Approximate GP Empirical Risk ({approximate_gp_gvi_loss_configuration}):",
                    f"{curve_function.__name__}",
                ]
            ),
            save_path=os.path.join(
                save_path,
                experiment_data.name,
                "tempered-approximate-gp-loss.png",
            ),
        )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    args = parser.parse_args()
    file_name = args.config_path.split("/")[-1].split(".")[0]
    print(args.config_path)
    print(args.action)
    print(file_name)

    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    if args.action == Actions.build_data.value:
        build_data_set(
            config=loaded_config, output_path=OUTPUT_PATH, experiment_name=file_name
        )
    elif args.action == Actions.train_reference.value:
        train_reference(
            config=loaded_config, output_path=OUTPUT_PATH, experiment_name=file_name
        )
    elif args.action == Actions.train_approximate.value:
        train_approximate(
            config=loaded_config, output_path=OUTPUT_PATH, experiment_name=file_name
        )
    elif args.action == Actions.temper_approximate.value:
        temper_approximate(
            config=loaded_config, output_path=OUTPUT_PATH, experiment_name=file_name
        )
    else:
        raise ValueError(f"Invalid action {args.action}")
