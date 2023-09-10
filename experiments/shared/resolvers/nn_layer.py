from typing import Dict, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.shared.schemas import NeuralNetworkLayerSchema


def nn_layer_resolver(
    x: jnp.ndarray,
    nn_layer_schema: NeuralNetworkLayerSchema,
    nn_layer_kwargs: Union[FrozenDict, Dict],
) -> jnp.ndarray:
    if nn_layer_schema == NeuralNetworkLayerSchema.convolution:
        assert "features" in nn_layer_kwargs, "Features must be specified."
        assert "kernel_size" in nn_layer_kwargs, "Kernel size must be specified."
        return nn.Conv(
            features=nn_layer_kwargs["features"],
            kernel_size=tuple(nn_layer_kwargs["kernel_size"]),
        )(x)
    if nn_layer_schema == NeuralNetworkLayerSchema.dense:
        assert "features" in nn_layer_kwargs, "Features must be specified."
        return nn.Dense(
            features=nn_layer_kwargs["features"],
        )(x)
    if nn_layer_schema == NeuralNetworkLayerSchema.average_pool:
        assert "window_shape" in nn_layer_kwargs, "Window shape must be specified."
        assert "strides" in nn_layer_kwargs, "Strides must be specified."
        return nn.avg_pool(
            x,
            window_shape=tuple(nn_layer_kwargs["window_shape"]),
            strides=tuple(nn_layer_kwargs["strides"]),
        )
    if nn_layer_schema == NeuralNetworkLayerSchema.relu:
        return nn.relu(x)
    if nn_layer_schema == NeuralNetworkLayerSchema.flatten:
        return x.reshape((x.shape[0], -1))
    if nn_layer_schema == NeuralNetworkLayerSchema.tanh:
        return nn.tanh(x)
    raise ValueError(f"Unknown neural network layer schema: {nn_layer_schema}.")
