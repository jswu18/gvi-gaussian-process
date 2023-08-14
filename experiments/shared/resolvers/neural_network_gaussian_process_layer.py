from typing import Callable, Dict, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from neural_tangents import stax

from experiments.shared.schemes import NeuralNetworkGaussianProcessLayerScheme


def neural_network_gaussian_process_layer_resolver(
    neural_network_gaussian_process_layer_scheme: NeuralNetworkGaussianProcessLayerScheme,
    neural_network_gaussian_process_layer_kwargs: Union[FrozenDict, Dict],
) -> Callable[[float, float], stax.layer]:
    if (
        neural_network_gaussian_process_layer_scheme
        == NeuralNetworkGaussianProcessLayerScheme.convolution
    ):
        assert (
            "features" in neural_network_gaussian_process_layer_kwargs
        ), "Features must be specified."
        assert (
            "kernel_size" in neural_network_gaussian_process_layer_kwargs
        ), "Kernel size must be specified."
        return lambda w_std, b_std: stax.Conv(
            out_chan=neural_network_gaussian_process_layer_kwargs["features"],
            filter_shape=neural_network_gaussian_process_layer_kwargs["kernel_size"],
            W_std=w_std,
            b_std=b_std,
            parameterization="standard",
        )
    elif (
        neural_network_gaussian_process_layer_scheme
        == NeuralNetworkGaussianProcessLayerScheme.dense
    ):
        assert (
            "features" in neural_network_gaussian_process_layer_kwargs
        ), "Features must be specified."
        return lambda w_std, b_std: stax.Dense(
            out_dim=neural_network_gaussian_process_layer_kwargs["features"],
            W_std=w_std,
            b_std=b_std,
            parameterization="standard",
        )
    elif (
        neural_network_gaussian_process_layer_scheme
        == NeuralNetworkGaussianProcessLayerScheme.average_pool
    ):
        assert (
            "window_shape" in neural_network_gaussian_process_layer_kwargs
        ), "Window shape must be specified."
        assert (
            "strides" in neural_network_gaussian_process_layer_kwargs
        ), "Strides must be specified."
        return lambda w_std, b_std: stax.AvgPool(
            window_shape=neural_network_gaussian_process_layer_kwargs["window_shape"],
            strides=neural_network_gaussian_process_layer_kwargs["strides"],
        )
    elif (
        neural_network_gaussian_process_layer_scheme
        == NeuralNetworkGaussianProcessLayerScheme.relu
    ):
        return stax.Relu()
    elif (
        neural_network_gaussian_process_layer_scheme
        == NeuralNetworkGaussianProcessLayerScheme.flatten
    ):
        return (stax.Flatten(),)
    elif (
        neural_network_gaussian_process_layer_scheme
        == NeuralNetworkGaussianProcessLayerScheme.tanh
    ):
        return nn.tanh(x)
    else:
        raise ValueError(
            f"Unknown neural network layer scheme: {neural_network_gaussian_process_layer_scheme}."
        )
