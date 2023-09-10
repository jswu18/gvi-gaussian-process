import argparse
import os

import pandas as pd

from experiments.shared.schemas import ActionSchema, ProblemSchema

parser = argparse.ArgumentParser(
    description="Main script for experiment config generation."
)
parser.add_argument(
    "--problem", choices=[ProblemSchema[a].value for a in ProblemSchema]
)


def concatenate_results(
    outputs_path: str,
) -> pd.DataFrame:
    df_list = []
    config_paths = [
        os.path.join(outputs_path, x)
        for x in os.listdir(outputs_path)
        if not os.path.isfile(os.path.join(outputs_path, x))
    ]
    for config_path in config_paths:
        dataset_paths = [
            os.path.join(config_path, x)
            for x in os.listdir(config_path)
            if not os.path.isfile(os.path.join(config_path, x))
        ]
        for dataset_path in dataset_paths:
            df_path = os.path.join(dataset_path, "metrics.csv")
            if os.path.isfile(df_path):
                df_list.append(pd.read_csv(df_path))
    return pd.concat(df_list, axis=0)


if __name__ == "__main__":
    args = parser.parse_args()

    base_outputs_path = f"experiments/{args.problem}/outputs"
    try:
        df_train_metrics = concatenate_results(
            outputs_path=os.path.join(
                base_outputs_path, str(ActionSchema.train_regulariser.value)
            ),
        )
        df_train_metrics.to_csv(
            os.path.join(
                base_outputs_path, f"{ActionSchema.train_regulariser.value}.csv"
            ),
            index=False,
        )
    except FileNotFoundError:
        pass

    try:
        df_train_metrics = concatenate_results(
            outputs_path=os.path.join(
                base_outputs_path, str(ActionSchema.train_approximate.value)
            ),
        )
        df_train_metrics.to_csv(
            os.path.join(
                base_outputs_path, f"{ActionSchema.train_approximate.value}.csv"
            ),
            index=False,
        )
    except FileNotFoundError:
        pass

    try:
        df_train_metrics = concatenate_results(
            outputs_path=os.path.join(
                base_outputs_path, str(ActionSchema.temper_approximate.value)
            ),
        )
        df_train_metrics.to_csv(
            os.path.join(
                base_outputs_path, f"{ActionSchema.temper_approximate.value}.csv"
            ),
            index=False,
        )
    except FileNotFoundError:
        pass
