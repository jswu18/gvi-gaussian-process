import enum


class DatasetSchema(str, enum.Enum):
    concrete = "concrete"
    energy = "energy"
    power = "power"
    wine = "wine"
    yacht = "yacht"
    boston = "boston"
    naval = "naval"
    kin8nm = "kin8nm"
    protein = "protein"
