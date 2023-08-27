from abc import ABC
from dataclasses import dataclass

from experiments.regression.schemas import DatasetSchema


@dataclass
class Dataset(ABC):
    input_column_names: list[str]
    output_column_name: str


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


class EnergyCoolingDataset(Dataset):
    input_column_names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    output_column_name = "Y2"


class EnergyHeatingDataset(Dataset):
    input_column_names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    output_column_name = "Y1"


class Kin8nmDataset(Dataset):
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


class NavalCompressorDataset(Dataset):
    input_column_names = [
        "Lever position",
        "Ship speed (v)",
        "GTT",
        "GTn",
        "GGn",
        "Ts",
        "Tp",
        "HP",
        "T1",
        "T2",
        "P48",
        "P1",
        "P2",
        "Pexh",
        "TIC",
        "mf",
    ]
    output_column_name = "Compressor DSC"


class NavalTurbineDataset(Dataset):
    input_column_names = [
        "Lever position",
        "Ship speed (v)",
        "GTT",
        "GTn",
        "GGn",
        "Ts",
        "Tp",
        "HP",
        "T1",
        "T2",
        "P48",
        "P1",
        "P2",
        "Pexh",
        "TIC",
        "mf",
    ]
    output_column_name = "Turbine DSC"


class PowerDataset(Dataset):
    input_column_names = ["AT", "V", "AP", "RH"]
    output_column_name = "PE"


class ProteinDataset(Dataset):
    input_column_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]
    output_column_name = "rmsd"


class RedWineDataset(Dataset):
    input_column_names = [
        "fixed" "acidity",
        "volatile" "acidity",
        "citric" "acid",
        "residual" "sugar",
        "chlorides",
        "free" "sulfur" "dioxide",
        "total" "sulfur" "dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    output_column_name = "quality"


class YachtDataset(Dataset):
    input_column_names = ["LC", "PC", "L/D", "B/Dr", "L/B", "Fr"]
    output_column_name = "Rr"


DATASET_SCHEMA_TO_DATASET = {
    DatasetSchema.boston: BostonDataset,
    DatasetSchema.concrete: ConcreteDataset,
    DatasetSchema.energy_cooling: EnergyCoolingDataset,
    DatasetSchema.energy_heating: EnergyHeatingDataset,
    DatasetSchema.kin8nm: Kin8nmDataset,
    DatasetSchema.naval_compressor: NavalCompressorDataset,
    DatasetSchema.naval_turbine: NavalTurbineDataset,
    DatasetSchema.power: PowerDataset,
    DatasetSchema.protein: ProteinDataset,
    DatasetSchema.red_wine: RedWineDataset,
    DatasetSchema.yacht: YachtDataset,
}
