from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.shared.nngp_kernels import CNNGPKernel, MLPGPKernel
from experiments.shared.schemes import NNGPKernelScheme


def custom_kernel_function_resolver(
    custom_kernel_function_scheme: NNGPKernelScheme,
    custom_kernel_function_kwargs: Union[FrozenDict, Dict],
) -> Union[Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.float64], Any]:
    if custom_kernel_function_scheme == NNGPKernelScheme.mlp:
        assert (
            "features" in custom_kernel_function_kwargs
        ), "Features must be specified."
        custom_kernel_function = MLPGPKernel(
            features=custom_kernel_function_kwargs["features"],
        )
        custom_kernel_function_parameters = (
            custom_kernel_function.initialise_parameters()
        )
        return custom_kernel_function, custom_kernel_function_parameters
    elif custom_kernel_function_scheme == NNGPKernelScheme.cnn:
        assert (
            "number_of_outputs" in custom_kernel_function_kwargs
        ), "Number of outputs must be specified."
        custom_kernel_function = CNNGPKernel(
            number_of_outputs=custom_kernel_function_kwargs["number_of_outputs"]
        )
        custom_kernel_function_parameters = (
            custom_kernel_function.initialise_parameters()
        )
        return custom_kernel_function, custom_kernel_function_parameters
    else:
        raise ValueError(
            f"Unknown custom kernel function scheme: {custom_kernel_function_scheme}."
        )
