import argparse
import os
from typing import Dict

import jax
import jax.numpy as jnp
import yaml

from experiments.regression.action_runners import build_approximate_gp
from experiments.regression.data import split_train_test_validation_data
from experiments.regression.metrics import calculate_metrics
from experiments.regression.plotters import plot_data, plot_prediction
from experiments.regression.trainers import meta_train_reference_gp
from experiments.shared.data import Data, ExperimentData
from experiments.shared.plotters import plot_losses, plot_two_losses
from experiments.shared.resolvers import (
    inducing_points_selector_resolver,
    kernel_resolver,
    trainer_settings_resolver,
)
from experiments.shared.resolvers.kernel import resolve_existing_kernel
from experiments.shared.schemas import ActionSchema, EmpiricalRiskSchema
from experiments.shared.trainers import train_approximate_gp, train_tempered_gp
from experiments.shared.utils import construct_path
from experiments.toy_curves.curves import CURVE_FUNCTIONS
from src.gps import GPRegression
from src.kernels import TemperedKernel
from src.means import ConstantMean

parser = argparse.ArgumentParser(description="Main script for toy curves experiments.")
parser.add_argument("--action", choices=[ActionSchema[a].value for a in ActionSchema])
parser.add_argument("--config_path", type=str)


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
        action=ActionSchema.build_data,
    )
    for curve_function in CURVE_FUNCTIONS:
        y = curve_function(
            key=jax.random.PRNGKey(config["seed"]),
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
            seed=jax.random.PRNGKey(config["seed"]),
            split_seed=jax.random.PRNGKey(curve_function.seed),
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
    for curve_function in CURVE_FUNCTIONS:
        experiment_data = ExperimentData.load(
            path=data_path,
            name=type(curve_function).__name__.lower(),
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
            data_dimension=experiment_data.train.x.shape[1],
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
        df_metrics = calculate_metrics(
            experiment_data=experiment_data,
            gp=reference_gp,
            gp_parameters=reference_gp_parameters,
            action=ActionSchema.train_reference,
            experiment_name=experiment_name,
            dataset_name=type(curve_function).__name__.lower(),
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


def train_approximate(config: Dict, output_path: str, experiment_name: str) -> None:
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
    config_directory_parent_path = os.path.dirname(os.path.abspath(__file__))
    reference_config_path = os.path.join(
        config_directory_parent_path,
        "configs",
        ActionSchema.train_reference,
        f"{config['reference_name']}.yaml",
    )
    with open(reference_config_path, "r") as reference_config_file:
        reference_config = yaml.safe_load(reference_config_file)
    data_path = construct_path(
        output_path=output_path,
        experiment_name=reference_config["data_name"],
        action=ActionSchema.build_data,
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
        (
            approximate_gp_parameters,
            approximate_post_epoch_history,
        ) = train_approximate_gp(
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
            dataset_name=type(curve_function).__name__.lower(),
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
            loss_name=f"{config['empirical_risk_schema']}+{config['regularisation']['regularisation_schema']}",
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
            loss2_name=config["regularisation"]["regularisation_schema"],
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
    config_directory_parent_path = os.path.dirname(os.path.abspath(__file__))
    approximate_path = construct_path(
        output_path=output_path,
        experiment_name=config["approximate_name"],
        action=ActionSchema.train_approximate,
    )
    approximate_config_path = os.path.join(
        config_directory_parent_path,
        "configs",
        ActionSchema.train_approximate,
        f"{config['approximate_name']}.yaml",
    )
    with open(approximate_config_path, "r") as approximate_config_file:
        approximate_config = yaml.safe_load(approximate_config_file)
    reference_config_path = os.path.join(
        config_directory_parent_path,
        "configs",
        ActionSchema.train_reference,
        f"{approximate_config['reference_name']}.yaml",
    )
    with open(reference_config_path, "r") as reference_config_file:
        reference_config = yaml.safe_load(reference_config_file)
    data_path = construct_path(
        output_path=output_path,
        experiment_name=reference_config["data_name"],
        action=ActionSchema.build_data,
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
                ActionSchema.train_approximate,
                f"{config['approximate_name']}.yaml",
            )
        )

        with open(approximate_config_path, "r") as approximate_config_file:
            approximate_config = yaml.safe_load(approximate_config_file)
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
        df_metrics = calculate_metrics(
            experiment_data=experiment_data,
            gp=tempered_gp,
            gp_parameters=tempered_gp_parameters,
            action=ActionSchema.temper_approximate,
            experiment_name=experiment_name,
            dataset_name=type(curve_function).__name__.lower(),
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

    # weird bug that needs this initialised to run fast on first iteration
    import matplotlib.pyplot as plt

    plt.subplots(figsize=(13, 6.5))

    print(args.config_path)

    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    if args.action == ActionSchema.build_data.value:
        build_data_set(
            config=loaded_config, output_path=OUTPUT_PATH, experiment_name=file_name
        )
    elif args.action == ActionSchema.train_reference.value:
        train_reference(
            config=loaded_config, output_path=OUTPUT_PATH, experiment_name=file_name
        )
    elif args.action == ActionSchema.train_approximate.value:
        train_approximate(
            config=loaded_config, output_path=OUTPUT_PATH, experiment_name=file_name
        )
    elif args.action == ActionSchema.temper_approximate.value:
        temper_approximate(
            config=loaded_config, output_path=OUTPUT_PATH, experiment_name=file_name
        )
    else:
        raise ValueError(f"Invalid action {args.action}")
