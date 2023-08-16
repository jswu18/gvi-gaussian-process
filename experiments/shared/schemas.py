import enum


class OptimiserSchema(str, enum.Enum):
    adam = "adam"
    adabelief = "adabelief"
    rmsprop = "rmsprop"


class EmpiricalRiskSchema(str, enum.Enum):
    negative_log_likelihood = "negative_log_likelihood"
    cross_entropy = "cross_entropy"


class RegularisationSchema(str, enum.Enum):
    point_wise_gaussian_wasserstein = "point_wise_gaussian_wasserstein"
    point_wise_kl = "point_wise_kl"
    point_wise_bhattacharyya = "point_wise_bhattacharyya"
    point_wise_hellinger = "point_wise_hellinger"
    point_wise_renyi = "point_wise_renyi"
    gaussian_squared_difference = "gaussian_squared_difference"
    gaussian_wasserstein = "gaussian_wasserstein"
    multinomial_wasserstein = "multinomial_wasserstein"


class KernelSchema(str, enum.Enum):
    polynomial = "polynomial"
    ard = "ard"
    custom = "custom"
    custom_mapping = "custom_mapping"
    neural_network = "neural_network"
    diagonal_svgp = "diagonal_svgp"
    kernelised_svgp = "kernelised_svgp"
    log_svgp = "log_svgp"
    svgp = "svgp"
    multi_output = "multi_output"
    custom_approximate = "custom_approximate"
    custom_mapping_approximate = "custom_mapping_approximate"


class MeanSchema(str, enum.Enum):
    constant = "constant"
    custom = "custom"
    multi_output = "multi_output"
    svgp = "svgp"


class NeuralNetworkLayerSchema(str, enum.Enum):
    convolution = "convolution"
    dense = "dense"
    average_pool = "average_pool"
    relu = "relu"
    tanh = "tanh"
    flatten = "flatten"


class NeuralNetworkGaussianProcessLayerSchema(str, enum.Enum):
    convolution = "convolution"
    dense = "dense"
    average_pool = "average_pool"
    relu = "relu"
    tanh = "tanh"
    flatten = "flatten"
