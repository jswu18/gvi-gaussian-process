import enum


class Optimiser(str, enum.Enum):
    adam = "adam"
    adabeleif = "adabelief"


class EmpiricalRiskScheme(str, enum.Enum):
    negative_log_likelihood = "negative_log_likelihood"


class RegularisationScheme(str, enum.Enum):
    squared_difference = "squared_difference"
    wasserstein = "wasserstein"
    point_wise_wasserstein = "point_wise_wasserstein"
    point_wise_kl = "point_wise_kl"
    point_wise_symmetric_kl = "point_wise_symmetric_kl"
    point_wise_bhattacaryya = "point_wise_bhattacaryya"
    point_wise_hellinger = "point_wise_hellinger"
    point_wise_renyi = "point_wise_renyi"
