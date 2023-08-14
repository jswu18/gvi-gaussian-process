from typing import Dict, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.shared.schemes import NeuralNetworkLayerScheme


def neural_network_layer_resolver(
    x: jnp.ndarray,
    neural_network_layer_scheme: NeuralNetworkLayerScheme,
    neural_network_layer_kwargs: Union[FrozenDict, Dict],
) -> jnp.ndarray:
    if neural_network_layer_scheme == NeuralNetworkLayerScheme.convolution:
        assert "features" in neural_network_layer_kwargs, "Features must be specified."
        assert (
            "kernel_size" in neural_network_layer_kwargs
        ), "Kernel size must be specified."
        return nn.Conv(
            features=neural_network_layer_kwargs["features"],
            kernel_size=neural_network_layer_kwargs["kernel_size"],
        )(x)
    elif neural_network_layer_scheme == NeuralNetworkLayerScheme.dense:
        assert "features" in neural_network_layer_kwargs, "Features must be specified."
        return nn.Dense(
            features=neural_network_layer_kwargs["features"],
        )(x)
    elif neural_network_layer_scheme == NeuralNetworkLayerScheme.average_pool:
        assert (
            "window_shape" in neural_network_layer_kwargs
        ), "Window shape must be specified."
        assert "strides" in neural_network_layer_kwargs, "Strides must be specified."
        return nn.avg_pool(
            x,
            window_shape=neural_network_layer_kwargs["window_shape"],
            strides=neural_network_layer_kwargs["strides"],
        )
    elif neural_network_layer_scheme == NeuralNetworkLayerScheme.relu:
        return nn.relu(x)
    elif neural_network_layer_scheme == NeuralNetworkLayerScheme.flatten:
        return x.reshape((x.shape[0], -1))
    elif neural_network_layer_scheme == NeuralNetworkLayerScheme.tanh:
        return nn.tanh(x)
    else:
        raise ValueError(
            f"Unknown neural network layer scheme: {neural_network_layer_scheme}."
        )
