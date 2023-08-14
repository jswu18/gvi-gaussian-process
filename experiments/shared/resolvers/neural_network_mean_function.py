from typing import Any, Callable, Dict, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.shared.resolvers import neural_network_layer_resolver


def neural_network_mean_function_resolver(
    neural_network_function_kwargs: Union[FrozenDict, Dict],
) -> Union[Callable[[Any, jnp.ndarray], jnp.ndarray], Any]:
    assert "layers" in neural_network_function_kwargs, "Layers must be specified."

    class NeuralNetwork(nn.Module):
        @nn.compact
        def __call__(self, x):
            for layer in neural_network_function_kwargs["layers"]:
                x = neural_network_layer_resolver(
                    x=x,
                    neural_network_layer_scheme=layer,
                    neural_network_layer_kwargs=neural_network_function_kwargs[
                        "layers"
                    ][layer],
                )
            return x.reshape(-1)

    neural_network = NeuralNetwork()

    assert "key" in neural_network_function_kwargs, "Key must be specified."
    assert (
        "input_shape" in neural_network_function_kwargs
    ), "Input shape must be specified."
    neural_network_parameters = neural_network.init(
        jax.random.PRNGKey(neural_network_function_kwargs["key"]),
        jnp.empty((1, *neural_network_function_kwargs["input_shape"])),
        train=False,
    )
    return (
        lambda parameters, x: neural_network.apply(parameters, x),
        neural_network_parameters,
    )
