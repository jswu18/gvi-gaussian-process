import argparse
import os

import jax
import yaml

from experiments.regression.utils import (
    build_data_set,
    temper_approximate,
    train_approximate,
    train_reference,
)
from experiments.shared.schemas import Actions

parser = argparse.ArgumentParser(description="Main script for toy curves experiments.")
parser.add_argument("--action", choices=[Actions[a].value for a in Actions])
parser.add_argument("--config_path", type=str)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    args = parser.parse_args()
    file_name = args.config_path.split("/")[-1].split(".")[0]
    # weird bug that needs this initialised to run fast on first iteration
    import matplotlib.pyplot as plt

    plt.subplots(figsize=(13, 6.5))
    data_csv_path = "experiments/regression/boston/data/data.csv"
    dataset_name = "boston"
    print(args.config_path)
    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    if args.action == Actions.build_data.value:
        build_data_set(
            config=loaded_config,
            data_csv_path=data_csv_path,
            output_path=OUTPUT_PATH,
            experiment_name=file_name,
            dataset_name=dataset_name,
            rescale_y=True,
        )
    elif args.action == Actions.train_reference.value:
        train_reference(
            config=loaded_config,
            output_path=OUTPUT_PATH,
            experiment_name=file_name,
            dataset_name=dataset_name,
        )
    elif args.action == Actions.train_approximate.value:
        train_approximate(
            config=loaded_config,
            output_path=OUTPUT_PATH,
            experiment_name=file_name,
            dataset_name=dataset_name,
            config_directory_parent_path=os.path.dirname(os.path.abspath(__file__)),
        )
    elif args.action == Actions.temper_approximate.value:
        temper_approximate(
            config=loaded_config,
            output_path=OUTPUT_PATH,
            experiment_name=file_name,
            config_directory_parent_path=os.path.dirname(os.path.abspath(__file__)),
            dataset_name=dataset_name,
        )
    else:
        raise ValueError(f"Invalid action {args.action}")
