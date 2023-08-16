from typing import Dict, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.shared.schemas import NeuralNetworkLayerScheme


def nn_layer_resolver(
    x: jnp.ndarray,
    nn_layer_scheme: NeuralNetworkLayerScheme,
    nn_layer_kwargs: Union[FrozenDict, Dict],
) -> jnp.ndarray:
    if nn_layer_scheme == NeuralNetworkLayerScheme.convolution:
        assert "features" in nn_layer_kwargs, "Features must be specified."
        assert "kernel_size" in nn_layer_kwargs, "Kernel size must be specified."
        return nn.Conv(
            features=nn_layer_kwargs["features"],
            kernel_size=tuple(nn_layer_kwargs["kernel_size"]),
        )(x)
    elif nn_layer_scheme == NeuralNetworkLayerScheme.dense:
        assert "features" in nn_layer_kwargs, "Features must be specified."
        return nn.Dense(
            features=nn_layer_kwargs["features"],
        )(x)
    elif nn_layer_scheme == NeuralNetworkLayerScheme.average_pool:
        assert "window_shape" in nn_layer_kwargs, "Window shape must be specified."
        assert "strides" in nn_layer_kwargs, "Strides must be specified."
        return nn.avg_pool(
            x,
            window_shape=tuple(nn_layer_kwargs["window_shape"]),
            strides=tuple(nn_layer_kwargs["strides"]),
        )
    elif nn_layer_scheme == NeuralNetworkLayerScheme.relu:
        return nn.relu(x)
    elif nn_layer_scheme == NeuralNetworkLayerScheme.flatten:
        return x.reshape((x.shape[0], -1))
    elif nn_layer_scheme == NeuralNetworkLayerScheme.tanh:
        return nn.tanh(x)
    else:
        raise ValueError(f"Unknown neural network layer scheme: {nn_layer_scheme}.")
