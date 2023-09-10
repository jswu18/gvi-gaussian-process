import argparse
import os
import shutil

from experiments.shared.schemas import ActionSchema, ProblemSchema

parser = argparse.ArgumentParser(
    description="Main script for experiment config generation."
)
parser.add_argument(
    "--problem",
    choices=[ProblemSchema[a].value for a in ProblemSchema],
    required=True,
)
parser.add_argument(
    "--build_data_max_runtime",
    type=str,
    required=False,
    default="01:00:00",
    help="Max runtime of build data jobs.",
)
parser.add_argument(
    "--train_regulariser_max_runtime",
    type=str,
    required=False,
    default="04:00:00",
    help="Max runtime of train regulariser jobs.",
)
parser.add_argument(
    "--train_approximate_max_runtime",
    type=str,
    required=False,
    default="04:00:00",
    help="Max runtime of train approximate jobs.",
)
parser.add_argument(
    "--temper_approximate_max_runtime",
    type=str,
    required=False,
    default="01:00:00",
    help="Max runtime of temper approximate jobs.",
)
parser.add_argument(
    "--build_data_num_gpus",
    type=int,
    required=False,
    default=0,
    help="Number of GPUs for build data jobs.",
)
parser.add_argument(
    "--train_regulariser_num_gpus",
    type=int,
    required=False,
    default=1,
    help="Number of GPUs for train regulariser jobs.",
)
parser.add_argument(
    "--train_approximate_num_gpus",
    type=int,
    required=False,
    default=1,
    help="Number of GPUs for train approximate jobs.",
)
parser.add_argument(
    "--temper_approximate_num_gpus",
    type=int,
    required=False,
    default=0,
    help="Number of GPUs for temper approximate jobs.",
)
parser.add_argument(
    "--build_data_num_cores",
    type=int,
    required=False,
    default=4,
    help="Number of cores for build data jobs.",
)
parser.add_argument(
    "--train_regulariser_num_cores",
    type=int,
    required=False,
    default=4,
    help="Number of cores for train regulariser jobs.",
)
parser.add_argument(
    "--train_approximate_num_cores",
    type=int,
    required=False,
    default=4,
    help="Number of cores for train approximate jobs.",
)
parser.add_argument(
    "--temper_approximate_num_cores",
    type=int,
    required=False,
    default=4,
    help="Number of cores for temper approximate jobs.",
)
parser.add_argument(
    "--build_data_mem",
    type=str,
    required=False,
    default="5G",
    help="Amount of RAM for build data jobs per core.",
)
parser.add_argument(
    "--train_regulariser_mem",
    type=str,
    required=False,
    default="5G",
    help="Amount of RAM for train regulariser jobs per core.",
)
parser.add_argument(
    "--train_approximate_mem",
    type=str,
    required=False,
    default="5G",
    help="Amount of RAM for train approximate jobs per core.",
)
parser.add_argument(
    "--temper_approximate_mem",
    type=str,
    required=False,
    default="5G",
    help="Amount of RAM for temper approximate jobs per core.",
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
        "# Max runtime of job. Shorter means scheduled faster.",
        f"#$ -l h_rt={max_runtime}",
        f"#$ -l gpu={num_gpus}",
        "# Number of cores. Set to more than one when using a GPU, but not too many.",
        f"#$ -pe smp {num_cores}",
        "# Amount of RAM.",
        "# IMPORTANT: each core gets RAM below, so in this case 20GB total.",
        f"#$ -l mem={mem}",
        "# Stops smaller jobs jumping in front of you in the queue.",
        "#$ -R y",
        "# Merges the standard output and error output files into one file.",
        "#$ -j y",
        "# Working directory of job.",
        "# (-cwd is short for set wd to the directory you run qsub)",
        "#$ -wd /home/ucabwuh/generalised-variational-inference-for-gaussian-processes",
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
    if os.path.exists(os.path.join(save_path, action.name)):
        shutil.rmtree(os.path.join(save_path, action.name))
    if not os.path.exists(os.path.join(save_path, action.name)):
        os.makedirs(os.path.join(save_path, action.name))
    shell_command_paths = [
        f
        for f in os.listdir(os.path.join(command_path, action.name))
        if os.path.isfile(os.path.join(command_path, action.name, f))
    ]
    number_of_commands = len(shell_command_paths)
    myriad_command_paths = []
    for i, shell_command_path in enumerate(shell_command_paths):
        if not shell_command_path.endswith(".sh"):
            continue
        base_commands = build_base_commands(
            job_name=f"{action.name}-{problem.name}-{i+1}-of-{number_of_commands}",
            max_runtime=max_runtime,
            num_gpus=num_gpus,
            num_cores=num_cores,
            mem=mem,
            repository_path=repository_path,
        )
        with open(
            os.path.join(command_path, action.name, shell_command_path), "r"
        ) as file:
            shell_command = file.read()
        myriad_command_path = os.path.join(save_path, action.name, f"{i}.sh")
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

    PRECOMPUTED_SHELL_COMMAND_PATH = f"experiments/{args.problem}/shell_commands"
    MYRIAD_SHELL_COMMAND_PATH = f"experiments/{args.problem}/myriad_shell_commands"
    generate_myriad_commands(
        problem=ProblemSchema(args.problem),
        action=ActionSchema.build_data,
        command_path=PRECOMPUTED_SHELL_COMMAND_PATH,
        save_path=MYRIAD_SHELL_COMMAND_PATH,
        max_runtime=args.build_data_max_runtime,
        num_gpus=args.build_data_num_gpus,
        num_cores=args.build_data_num_cores,
        mem=args.build_data_mem,
        repository_path=os.getcwd(),
    )
    generate_myriad_commands(
        problem=ProblemSchema(args.problem),
        action=ActionSchema.train_regulariser,
        command_path=PRECOMPUTED_SHELL_COMMAND_PATH,
        save_path=MYRIAD_SHELL_COMMAND_PATH,
        max_runtime=args.train_regulariser_max_runtime,
        num_gpus=args.train_regulariser_num_gpus,
        num_cores=args.train_regulariser_num_cores,
        mem=args.train_regulariser_mem,
        repository_path=os.getcwd(),
    )
    generate_myriad_commands(
        problem=ProblemSchema(args.problem),
        action=ActionSchema.train_approximate,
        command_path=PRECOMPUTED_SHELL_COMMAND_PATH,
        save_path=MYRIAD_SHELL_COMMAND_PATH,
        max_runtime=args.train_approximate_max_runtime,
        num_gpus=args.train_approximate_num_gpus,
        num_cores=args.train_approximate_num_cores,
        mem=args.train_approximate_mem,
        repository_path=os.getcwd(),
    )
    generate_myriad_commands(
        problem=ProblemSchema(args.problem),
        action=ActionSchema.temper_approximate,
        command_path=PRECOMPUTED_SHELL_COMMAND_PATH,
        save_path=MYRIAD_SHELL_COMMAND_PATH,
        max_runtime=args.temper_approximate_max_runtime,
        num_gpus=args.temper_approximate_num_gpus,
        num_cores=args.temper_approximate_num_cores,
        mem=args.temper_approximate_mem,
        repository_path=os.getcwd(),
    )
