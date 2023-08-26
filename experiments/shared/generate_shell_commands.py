import os
import shutil

import jax.numpy as jnp
import pandas as pd

from experiments.shared.schemas import Actions


def generate_shell_commands(
    action: Actions,
    repository_path: str,
    main_path: str,
    config_path: str,
    save_path: str,
    chunk_size: int,
):
    config_path = os.path.join(config_path, action)
    df = pd.read_csv(f"{config_path}.csv")
    commands = [
        f"python {main_path} --action {action} --config_path {os.path.join(config_path, uuid_identifier)}.yaml"
        for uuid_identifier in df.uuid
    ]
    if os.path.exists(os.path.join(save_path, action)):
        shutil.rmtree(os.path.join(save_path, action))
    if not os.path.exists(os.path.join(save_path, action)):
        os.makedirs(os.path.join(save_path, action))
    for i in range(0, int(jnp.ceil(len(commands) / chunk_size))):
        with open(os.path.join(save_path, action, f"{i}.sh"), "w") as file:
            file.write(
                "\n".join(
                    [f"cd {repository_path}", "export PYTHONPATH=$PWD"]
                    + commands[i * chunk_size : (i + 1) * chunk_size]
                )
            )


if __name__ == "__main__":
    problem_type = "regression"
    dataset_name = "boston"
    main_path = f"experiments/{problem_type}/{dataset_name}/main.py"
    config_path = f"experiments/{problem_type}/{dataset_name}/configs"
    save_path = f"experiments/{problem_type}/{dataset_name}/shell_commands"
    generate_shell_commands(
        action=Actions.build_data,
        repository_path=os.getcwd(),
        main_path=main_path,
        config_path=config_path,
        save_path=save_path,
        chunk_size=1000,
    )
    generate_shell_commands(
        action=Actions.train_reference,
        repository_path=os.getcwd(),
        main_path=main_path,
        config_path=config_path,
        save_path=save_path,
        chunk_size=1000,
    )
    generate_shell_commands(
        action=Actions.train_approximate,
        repository_path=os.getcwd(),
        main_path=main_path,
        config_path=config_path,
        save_path=save_path,
        chunk_size=1000,
    )
    generate_shell_commands(
        action=Actions.temper_approximate,
        repository_path=os.getcwd(),
        main_path=main_path,
        config_path=config_path,
        save_path=save_path,
        chunk_size=1000,
    )
