import enum


class DatasetSchema(str, enum.Enum):
    boston = "boston"
    concrete = "concrete"
    energy_cooling = "energy_cooling"
    energy_heating = "energy_heating"
    kin8nm = "kin8nm"
    naval_compressor = "naval_compressor"
    naval_turbine = "naval_turbine"
    power = "power"
    protein = "protein"
    red_wine = "red_wine"
    yacht = "yacht"
