import os
import shutil
import uuid
from typing import Dict, Optional

import pandas as pd
import yaml


def merge_dictionaries(dict1: Dict, dict2: Dict) -> Dict:
    # https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    """merges b into a"""
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield k, dict(merge_dictionaries(dict1[k], dict2[k]))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield k, dict2[k]
                # Alternatively, replace this with exception raiser to alert you of value conflicts
        elif k in dict1:
            yield k, dict1[k]
        else:
            yield k, dict2[k]


def generate_configs(
    config_path: str, output_path: str, reference_configs: Optional[Dict] = None
) -> None:
    with open(os.path.join(config_path, "base.yaml"), "r") as file:
        base_config = yaml.safe_load(file)

    folders = [
        f
        for f in os.listdir(config_path)
        if not os.path.isfile(os.path.join(config_path, f))
    ]
    configs = [base_config]
    df_configs = [{}]
    for folder in folders:
        file_paths = [
            f
            for f in os.listdir(os.path.join(config_path, folder))
            if os.path.isfile(os.path.join(config_path, folder, f))
        ]
        temp = []
        df_temp = []
        for f in file_paths:
            with open(os.path.join(config_path, folder, f), "r") as file:
                config = yaml.safe_load(file)
            temp.extend([dict(merge_dictionaries(x, config)) for x in configs])
            df_temp.extend(
                [
                    dict(merge_dictionaries(x, {folder: f.split(".")[0]}))
                    for x in df_configs
                ]
            )
        configs = temp
        df_configs = df_temp
    if reference_configs:
        temp = []
        df_temp = []
        for reference_name in reference_configs:
            df = pd.read_csv(f"{reference_configs[reference_name]}.csv")
            for uuid_identifier in df.uuid:
                temp.extend(
                    [
                        dict(merge_dictionaries(x, {reference_name: uuid_identifier}))
                        for x in configs
                    ]
                )
                df_temp.extend(
                    [
                        dict(merge_dictionaries(x, {reference_name: uuid_identifier}))
                        for x in df_configs
                    ]
                )
        configs = temp
        df_configs = df_temp
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, config in enumerate(configs):
        uuid_identifier = uuid.uuid4().hex
        with open(os.path.join(output_path, f"{uuid_identifier}.yaml"), "w") as file:
            yaml.dump(config, file)
        df_configs[i]["uuid"] = uuid_identifier
    df = pd.DataFrame(df_configs)
    df.set_index("uuid", inplace=True)
    df.sort_values(list(df.columns), inplace=True)
    df.to_csv(f"{output_path}.csv")


if __name__ == "__main__":
    generate_configs(
        config_path="experiments/regression/toy_curves/base_configs/build_data",
        output_path="experiments/regression/toy_curves/configs/build_data",
    )
    generate_configs(
        config_path="experiments/regression/toy_curves/base_configs/train_reference",
        output_path="experiments/regression/toy_curves/configs/train_reference",
        reference_configs={
            "data_name": "experiments/regression/toy_curves/configs/build_data"
        },
    )
    generate_configs(
        config_path="experiments/regression/toy_curves/base_configs/train_approximate",
        output_path="experiments/regression/toy_curves/configs/train_approximate",
        reference_configs={
            "reference_name": "experiments/regression/toy_curves/configs/train_reference"
        },
    )
    generate_configs(
        config_path="experiments/regression/toy_curves/base_configs/temper_approximate",
        output_path="experiments/regression/toy_curves/configs/temper_approximate",
        reference_configs={
            "approximate_name": "experiments/regression/toy_curves/configs/train_approximate"
        },
    )
