from typing import Any, Callable, Dict, List, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from neural_tangents import stax

from experiments.shared.resolvers.nngp_layer import nngp_layer_resolver


def nngp_kernel_function_resolver(
    nngp_kernel_function_kwargs: Union[FrozenDict, Dict],
) -> Tuple[Callable[[Any, jnp.ndarray], jnp.ndarray], Dict]:
    assert "layers" in nngp_kernel_function_kwargs, "Layers must be specified."
    nn_layers = []
    is_parameterised_array = [False] * len(nngp_kernel_function_kwargs["layers"])
    is_parameterised_count = 0
    for i, layer in enumerate(nngp_kernel_function_kwargs["layers"]):
        layer_params = nngp_kernel_function_kwargs["layers"][layer]
        assert (
            "layer_schema" in layer_params
        ), f"Layer schema must be specified for {layer=}."
        assert (
            "layer_kwargs" in layer_params
        ), f"Layer kwargs must be specified for {layer=}."
        nn_layer, is_parameterised = nngp_layer_resolver(
            nngp_layer_schema=layer_params["layer_schema"],
            nngp_layer_kwargs=layer_params["layer_kwargs"],
        )
        if is_parameterised:
            is_parameterised_count += 1
        nn_layers.append(nn_layer)
        is_parameterised_array[i] = is_parameterised

    def kernel_function(parameters, x1, x2, nn_layers_, is_parameterised_array_):
        nn_architecture = []
        for i_, nn_layer_ in enumerate(nn_layers_):
            if is_parameterised_array_[i_]:
                nn_architecture.append(
                    nn_layer_(parameters["w_std"][i_], parameters["b_std"][i_])
                )
            else:
                nn_architecture.append(nn_layer_)
        _, _, kernel_fn = stax.serial(*nn_architecture)
        return kernel_fn(x1, x2, "nngp")

    init_parameters = {
        "w_std": jnp.ones((is_parameterised_count,)),
        "b_std": jnp.ones((is_parameterised_count,)),
    }

    return (
        lambda parameters, x1, x2: kernel_function(
            parameters, x1, x2, nn_layers, is_parameterised_array
        ),
        init_parameters,
    )
