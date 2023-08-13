import enum


class OptimiserScheme(str, enum.Enum):
    adam = "adam"
    adabelief = "adabelief"
    rmsprop = "rmsprop"


class EmpiricalRiskScheme(str, enum.Enum):
    negative_log_likelihood = "negative_log_likelihood"
    cross_entropy = "cross_entropy"


class RegularisationScheme(str, enum.Enum):
    point_wise_gaussian_wasserstein = "point_wise_gaussian_wasserstein"
    point_wise_kl = "point_wise_kl"
    point_wise_symmetric_kl = "point_wise_symmetric_kl"
    point_wise_bhattacharyya = "point_wise_bhattacharyya"
    point_wise_hellinger = "point_wise_hellinger"
    point_wise_renyi = "point_wise_renyi"
    gaussian_squared_difference = "gaussian_squared_difference"
    gaussian_wasserstein = "gaussian_wasserstein"
    multinomial_wasserstein = "multinomial_wasserstein"
