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
    nngp = "nngp"
    custom_mapping = "custom_mapping"
    neural_network = "neural_network"
    diagonal_svgp = "diagonal_svgp"
    kernelised_svgp = "kernelised_svgp"
    log_svgp = "log_svgp"
    decomposed_svgp = "decomposed_svgp"
    multi_output = "multi_output"
    sparse_posterior = "sparse_posterior"
    fixed_sparse_posterior = "fixed_sparse_posterior"
    inner_product = "inner_product"


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


class InducingPointsSelectorSchema(str, enum.Enum):
    random = "random"
    conditional_variance = "conditional_variance"


class Actions(str, enum.Enum):
    build_data = "build_data"
    train_reference = "train_reference"
    train_approximate = "train_approximate"
    temper_approximate = "temper_approximate"
