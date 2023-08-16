from typing import Callable, Dict, Tuple, Union

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from neural_tangents import stax

from experiments.shared.schemas import NeuralNetworkGaussianProcessLayerSchema


def nngp_layer_resolver(
    nngp_layer_schema: NeuralNetworkGaussianProcessLayerSchema,
    nngp_layer_kwargs: Union[FrozenDict, Dict],
) -> Tuple[
    Union[
        Callable[[jnp.float64, jnp.float64], stax.layer],
        Tuple[Callable, Callable, Callable],
    ],
    bool,
]:
    if nngp_layer_schema == NeuralNetworkGaussianProcessLayerSchema.convolution:
        assert "features" in nngp_layer_kwargs, "Features must be specified."
        assert "kernel_size" in nngp_layer_kwargs, "Kernel size must be specified."
        return (
            lambda w_std, b_std: stax.Conv(
                out_chan=nngp_layer_kwargs["features"],
                filter_shape=tuple(nngp_layer_kwargs["kernel_size"]),
                W_std=w_std,
                b_std=b_std,
                parameterization="standard",
            ),
            True,
        )
    elif nngp_layer_schema == NeuralNetworkGaussianProcessLayerSchema.dense:
        assert "features" in nngp_layer_kwargs, "Features must be specified."
        return (
            lambda w_std, b_std: stax.Dense(
                out_dim=nngp_layer_kwargs["features"],
                W_std=w_std,
                b_std=b_std,
                parameterization="standard",
            ),
            True,
        )
    elif nngp_layer_schema == NeuralNetworkGaussianProcessLayerSchema.average_pool:
        assert "window_shape" in nngp_layer_kwargs, "Window shape must be specified."
        assert "strides" in nngp_layer_kwargs, "Strides must be specified."
        return (
            stax.AvgPool(
                window_shape=tuple(nngp_layer_kwargs["window_shape"]),
                strides=tuple(nngp_layer_kwargs["strides"]),
            ),
            False,
        )
    elif nngp_layer_schema == NeuralNetworkGaussianProcessLayerSchema.relu:
        return stax.Relu(), False
    elif nngp_layer_schema == NeuralNetworkGaussianProcessLayerSchema.flatten:
        return stax.Flatten(), False
    elif nngp_layer_schema == NeuralNetworkGaussianProcessLayerSchema.tanh:
        return stax.Erf(), False
    else:
        raise ValueError(f"Unknown neural network layer schema: {nngp_layer_schema}.")
