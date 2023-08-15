import argparse
import enum
import os
from typing import Dict

import jax
import jax.numpy as jnp
import yaml

from experiments.regression.data import split_train_test_validation_data
from experiments.regression.plotters import plot_data, plot_prediction
from experiments.regression.toy_curves.curves import CURVE_FUNCTIONS
from experiments.regression.trainers import meta_train_reference_gp
from experiments.shared.data import Data, ExperimentData
from experiments.shared.plotters import plot_losses
from experiments.shared.resolvers import kernel_resolver, trainer_settings_resolver


class Actions(str, enum.Enum):
    build_data = "build_data"
    train_reference = "train_reference"
    train_approximate = "train_approximate"
    temper_approximate = "temper_approximate"


parser = argparse.ArgumentParser(description="Main script for toy curves experiments.")
parser.add_argument("--action", choices=[Actions[a].value for a in Actions])
parser.add_argument("--config_path", type=str)


def construct_data_path(output_path: str, experiment_name: str) -> str:
    return os.path.join(
        output_path,
        "data",
        experiment_name,
    )


def construct_train_reference_path(output_path: str, experiment_name: str) -> str:
    return os.path.join(
        output_path,
        "reference",
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
    data_path = construct_data_path(
        output_path=output_path, experiment_name=experiment_name
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
        "empirical_risk_scheme" in config
    ), "Empirical risk scheme must be specified for reference training"
    assert (
        "empirical_risk_break_condition" in config
    ), "Empirical risk break condition must be specified for reference training"
    assert (
        "save_checkpoint_frequency" in config
    ), "Save checkpoint frequency must be specified for reference training"

    save_path = construct_train_reference_path(
        output_path=output_path, experiment_name=experiment_name
    )
    data_path = construct_data_path(
        output_path=output_path, experiment_name=config["data_name"]
    )
    for curve_function in CURVE_FUNCTIONS:
        experiment_data = ExperimentData.load(
            path=data_path,
            name=type(curve_function).__name__.lower(),
        )
        number_of_inducing_points = int(
            config["inducing_points_factor"] * jnp.sqrt(len(experiment_data.train.x))
        )
        trainer_settings = trainer_settings_resolver(
            trainer_settings_config=config["trainer_settings"],
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
            empirical_risk_scheme=config["empirical_risk_scheme"],
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
            loss_name=reference_gp_empirical_risk_scheme.value,
            title=f"Reference GP Empirical Risk: {curve_function.__name__}",
            save_path=os.path.join(
                curve_directory,
                "reference-gp-losses.png",
            ),
        )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    OUTPUT_PATH = "outputs"
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
        pass
    elif args.action == Actions.temper_approximate.value:
        pass
