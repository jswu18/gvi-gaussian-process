from typing import Any, Callable, Dict, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.shared.resolvers.nn_layer import nn_layer_resolver


def nn_function_resolver(
    nn_function_kwargs: Union[FrozenDict, Dict],
    data_dimension: int,
) -> Tuple[Callable, Any]:
    assert "layers" in nn_function_kwargs, "Layers must be specified."

    class NeuralNetwork(nn.Module):
        @nn.compact
        def __call__(self, x):
            for layer in nn_function_kwargs["layers"]:
                layer_params = nn_function_kwargs["layers"][layer]
                assert (
                    "layer_schema" in layer_params
                ), f"Layer schema must be specified for {layer=}."
                assert (
                    "layer_kwargs" in layer_params
                ), f"Layer kwargs must be specified for {layer=}."
                x = nn_layer_resolver(
                    x=x,
                    nn_layer_schema=layer_params["layer_schema"],
                    nn_layer_kwargs=layer_params["layer_kwargs"],
                )
            return x.reshape(-1)

    neural_network = NeuralNetwork()

    assert (
        "seed" in nn_function_kwargs
    ), "Seed must be specified for parameter initialisation."
    fake_input = (
        jnp.empty((1, *nn_function_kwargs["input_shape"]))
        if "input_shape" in nn_function_kwargs
        else jnp.empty((1, data_dimension))
    )
    neural_network_parameters = neural_network.init(
        jax.random.PRNGKey(nn_function_kwargs["seed"]),
        fake_input,
    )
    return (
        jax.jit(lambda parameters, x: neural_network.apply(parameters, x)),
        neural_network_parameters,
    )
