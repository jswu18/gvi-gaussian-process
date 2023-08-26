import argparse
import os
import shutil

from experiments.shared.schemas import ActionSchema, ProblemSchema

parser = argparse.ArgumentParser(
    description="Main script for experiment config generation."
)
parser.add_argument(
    "--problem", choices=[ProblemSchema[a].value for a in ProblemSchema]
)


def build_base_commands(
    job_name: str,
    max_runtime: str,
    num_gpus: int,
    num_cores: int,
    mem: str,
    repository_path: str,
):
    base_commands = [
        "#$ -S /bin/bash",
        f"# Max runtime of job. Shorter means scheduled faster.",
        f"#$ -l h_rt={max_runtime}",
        f"#$ -l gpu={num_gpus}",
        f"# Number of cores. Set to more than one when using a GPU, but not too many.",
        f"#$ -pe smp {num_cores}",
        f"# Amount of RAM.",
        f"# IMPORTANT: each core gets RAM below, so in this case 20GB total.",
        f"#$ -l mem={mem}",
        f"# Stops smaller jobs jumping in front of you in the queue.",
        "#$ -R y",
        "# Merges the standard output and error output files into one file.",
        "#$ -j y",
        "# Working directory of job.",
        "# (-cwd is short for set wd to the directory you run qsub)",
        f"#$ -wd /home/ucabwuh/generalised-variational-inference-for-gaussian-processes",
        "# Name of the job",
        f"#$ -N {job_name}",
        "date",
        "nvidia-smi",
        "module load python3/3.11",
        f"cd {repository_path}",
        "source .venv/bin/activate",
    ]
    return base_commands


def generate_myriad_commands(
    action: ActionSchema,
    repository_path: str,
    problem: ProblemSchema,
    command_path: str,
    save_path: str,
    max_runtime: str,
    num_gpus: int,
    num_cores: int,
    mem: str,
):
    if os.path.exists(os.path.join(save_path, action)):
        shutil.rmtree(os.path.join(save_path, action))
    if not os.path.exists(os.path.join(save_path, action)):
        os.makedirs(os.path.join(save_path, action))
    shell_command_paths = [
        f
        for f in os.listdir(os.path.join(command_path, action))
        if os.path.isfile(os.path.join(command_path, action, f))
    ]
    number_of_commands = len(shell_command_paths)
    myriad_command_paths = []
    for i, shell_command_path in enumerate(shell_command_paths):
        base_commands = build_base_commands(
            job_name=f"{problem.name}-{action.name}-{i+1}-of-{number_of_commands}",
            max_runtime=max_runtime,
            num_gpus=num_gpus,
            num_cores=num_cores,
            mem=mem,
            repository_path=repository_path,
        )
        with open(os.path.join(command_path, action, shell_command_path), "r") as file:
            shell_command = file.read()
        myriad_command_path = os.path.join(save_path, action, f"{i}.sh")
        with open(myriad_command_path, "w") as file:
            file.write("\n".join(base_commands + [shell_command]))
        myriad_command_paths.append(myriad_command_path)
    with open(f"{problem.name}-{action.name}.sh", "w") as file:
        file.write(
            "\n".join(
                [
                    "qsub " + myriad_command_path
                    for myriad_command_path in myriad_command_paths
                ]
            )
        )


if __name__ == "__main__":
    args = parser.parse_args()

    precomputed_shell_command_path = f"experiments/{args.problem}/shell_commands"
    myriad_shell_command_path = f"experiments/{args.problem}/myriad_shell_commands"
    generate_myriad_commands(
        problem=ProblemSchema(args.problem),
        action=ActionSchema.build_data,
        command_path=precomputed_shell_command_path,
        save_path=myriad_shell_command_path,
        max_runtime="01:00:00",
        num_gpus=0,
        num_cores=4,
        mem="5G",
        repository_path=os.getcwd(),
    )
    generate_myriad_commands(
        problem=ProblemSchema(args.problem),
        action=ActionSchema.train_reference,
        command_path=precomputed_shell_command_path,
        save_path=myriad_shell_command_path,
        max_runtime="04:00:00",
        num_gpus=1,
        num_cores=4,
        mem="5G",
        repository_path=os.getcwd(),
    )
    generate_myriad_commands(
        problem=ProblemSchema(args.problem),
        action=ActionSchema.train_approximate,
        command_path=precomputed_shell_command_path,
        save_path=myriad_shell_command_path,
        max_runtime="04:00:00",
        num_gpus=1,
        num_cores=4,
        mem="5G",
        repository_path=os.getcwd(),
    )
    generate_myriad_commands(
        problem=ProblemSchema(args.problem),
        action=ActionSchema.temper_approximate,
        command_path=precomputed_shell_command_path,
        save_path=myriad_shell_command_path,
        max_runtime="01:00:00",
        num_gpus=0,
        num_cores=4,
        mem="5G",
        repository_path=os.getcwd(),
    )
