from abc import ABC
from dataclasses import dataclass

from experiments.regression.schemas import DatasetSchema


@dataclass
class Dataset(ABC):
    input_column_names: list[str]
    output_column_name: str


class ConcreteDataset(Dataset):
    input_column_names = [
        "cement",
        "blast_furnace_slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_aggregate",
        "fine_aggregate",
        "age",
    ]
    output_column_name = "concrete_compressive_strength"


class BostonDataset(Dataset):
    input_column_names = [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
    ]
    output_column_name = "medv"


class ProteinDataset(Dataset):
    input_column_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]
    output_column_name = "rmsd"


class kin8nmDataset(Dataset):
    input_column_names = [
        "theta1",
        "theta2",
        "theta3",
        "theta4",
        "theta5",
        "theta6",
        "theta7",
        "theta8",
    ]
    output_column_name = "y"


DATASET_SCHEMA_TO_DATASET = {
    DatasetSchema.concrete: ConcreteDataset,
    DatasetSchema.boston: BostonDataset,
    DatasetSchema.protein: ProteinDataset,
    DatasetSchema.kin8nm: kin8nmDataset,
}
