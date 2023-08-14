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
    point_wise_bhattacharyya = "point_wise_bhattacharyya"
    point_wise_hellinger = "point_wise_hellinger"
    point_wise_renyi = "point_wise_renyi"
    gaussian_squared_difference = "gaussian_squared_difference"
    gaussian_wasserstein = "gaussian_wasserstein"
    multinomial_wasserstein = "multinomial_wasserstein"


class KernelScheme(str, enum.Enum):
    polynomial = "polynomial"
    ard = "ard"
    custom = "custom"
    neural_network = "neural_network"
    diagonal_svgp = "diagonal_svgp"
    kernelised_svgp = "kernelised_svgp"
    log_svgp = "log_svgp"
    svgp = "svgp"
    multi_output = "multi_output"


class MeanScheme(str, enum.Enum):
    constant = "constant"
    custom = "custom"
    multi_output = "multi_output"
    svgp = "svgp"


class NeuralNetworkLayerScheme(str, enum.Enum):
    convolution = "convolution"
    dense = "dense"
    average_pool = "average_pool"
    relu = "relu"
    tanh = "tanh"
    flatten = "flatten"


class NeuralNetworkGaussianProcessLayerScheme(str, enum.Enum):
    convolution = "convolution"
    dense = "dense"
    average_pool = "average_pool"
    relu = "relu"
    tanh = "tanh"
    flatten = "flatten"
