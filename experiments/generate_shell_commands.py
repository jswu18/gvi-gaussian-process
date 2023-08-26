import argparse
import os
import shutil

import jax.numpy as jnp
import pandas as pd

from experiments.shared.schemas import ActionSchema, ProblemSchema

parser = argparse.ArgumentParser(
    description="Main script for experiment config generation."
)
parser.add_argument(
    "--problem", choices=[ProblemSchema[a].value for a in ProblemSchema], required=True
)
parser.add_argument(
    "--train_reference_chunk_size",
    type=int,
    default=50,
    required=False,
    help="Number of configs per train reference shell file.",
)
parser.add_argument(
    "--train_approximate_chunk_size",
    type=int,
    default=50,
    required=False,
    help="Number of configs per train reference shell file.",
)
parser.add_argument(
    "--temper_approximate_chunk_size",
    type=int,
    default=100,
    required=False,
    help="Number of configs per temper shell file.",
)
parser.add_argument(
    "--build_data_chunk_size",
    type=int,
    default=100,
    required=False,
    help="Number of configs per build data shell file.",
)


def generate_shell_commands(
    action: ActionSchema,
    repository_path: str,
    main_path: str,
    config_path: str,
    save_path: str,
    chunk_size: int,
):
    config_path = os.path.join(config_path, action.name)
    df = pd.read_csv(f"{config_path}.csv")
    commands = [
        f"python {main_path} --action {action.name} --config_path {os.path.join(config_path, uuid_identifier)}.yaml"
        for uuid_identifier in df.uuid
    ]
    if os.path.exists(os.path.join(save_path, action.name)):
        shutil.rmtree(os.path.join(save_path, action.name))
    if not os.path.exists(os.path.join(save_path, action.name)):
        os.makedirs(os.path.join(save_path, action.name))
    for i in range(0, int(jnp.ceil(len(commands) / chunk_size))):
        with open(os.path.join(save_path, action.name, f"{i}.sh"), "w") as file:
            file.write(
                "\n".join(
                    [f"cd {repository_path}", "export PYTHONPATH=$PWD"]
                    + commands[i * chunk_size : (i + 1) * chunk_size]
                )
            )


if __name__ == "__main__":
    args = parser.parse_args()

    main_path = f"experiments/{args.problem}/main.py"
    config_path = f"experiments/{args.problem}/configs"
    save_path = f"experiments/{args.problem}/shell_commands"
    generate_shell_commands(
        action=ActionSchema.build_data,
        repository_path=os.getcwd(),
        main_path=main_path,
        config_path=config_path,
        save_path=save_path,
        chunk_size=args.build_data_chunk_size,
    )
    generate_shell_commands(
        action=ActionSchema.train_reference,
        repository_path=os.getcwd(),
        main_path=main_path,
        config_path=config_path,
        save_path=save_path,
        chunk_size=args.train_reference_chunk_size,
    )
    generate_shell_commands(
        action=ActionSchema.train_approximate,
        repository_path=os.getcwd(),
        main_path=main_path,
        config_path=config_path,
        save_path=save_path,
        chunk_size=args.train_approximate_chunk_size,
    )
    generate_shell_commands(
        action=ActionSchema.temper_approximate,
        repository_path=os.getcwd(),
        main_path=main_path,
        config_path=config_path,
        save_path=save_path,
        chunk_size=args.temper_approximate_chunk_size,
    )
