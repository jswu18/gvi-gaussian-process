from typing import Dict, Tuple, Union

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from experiments.shared.resolvers import custom_kernel_function_resolver
from experiments.shared.schemes import KernelScheme
from src.kernels import CustomKernel, MultiOutputKernel
from src.kernels.base import KernelBase, KernelBaseParameters
from src.kernels.non_stationary import PolynomialKernel
from src.kernels.svgp import (
    DiagonalSVGPKernel,
    KernelisedSVGPKernel,
    LogSVGPKernel,
    SVGPKernel,
)


def kernel_resolver(
    kernel_scheme: KernelScheme,
    kernel_kwargs: Union[FrozenDict, Dict],
    kernel_parameters: Union[FrozenDict, Dict],
) -> Tuple[KernelBase, KernelBaseParameters]:
    if kernel_scheme == KernelScheme.polynomial:
        assert (
            "polynomial_degree" in kernel_kwargs
        ), "Polynomial degree must be specified."
        kernel = PolynomialKernel(
            polynomial_degree=kernel_kwargs["polynomial_degree"],
        )

        assert "log_constant" in kernel_parameters, "Log constant must be specified."
        assert "log_scaling" in kernel_parameters, "Log scaling must be specified."
        kernel_parameters = kernel.generate_parameters(
            {
                "log_constant": kernel_parameters["log_constant"],
                "log_scaling": kernel_parameters["log_scaling"],
            }
        )
        return kernel, kernel_parameters
    elif kernel_scheme == KernelScheme.multi_output:
        if "repeat" in kernel_kwargs:
            kernel_list = []
            kernel_parameters_list = []
            for _ in range(kernel_kwargs["repeat"]):
                kernel, kernel_parameters = kernel_resolver(
                    kernel_scheme=kernel_kwargs["kernel_scheme"],
                    kernel_kwargs=kernel_kwargs["kernel_kwargs"],
                    kernel_parameters=kernel_parameters,
                )
                kernel_list.append(kernel)
                kernel_parameters_list.append(kernel_parameters)
            kernel = MultiOutputKernel(kernels=kernel_list)
            kernel_parameters = kernel.generate_parameters(
                {"kernels": kernel_parameters_list}
            )
            return kernel, kernel_parameters
        else:
            kernel_list = []
            kernel_parameters_list = []
            for kernel_kwargs_, kernel_parameters_ in zip(
                kernel_kwargs, kernel_parameters
            ):
                kernel, kernel_parameters = kernel_resolver(
                    kernel_scheme=kernel_kwargs[kernel_kwargs_]["kernel_scheme"],
                    kernel_kwargs=kernel_kwargs[kernel_kwargs_]["kernel_kwargs"],
                    kernel_parameters=kernel_parameters[kernel_parameters_][
                        "kernel_parameters"
                    ],
                )
                kernel_list.append(kernel)
                kernel_parameters_list.append(kernel_parameters)
            kernel = MultiOutputKernel(kernels=kernel_list)
            kernel_parameters = kernel.generate_parameters(
                {"kernels": kernel_parameters_list}
            )
            return kernel, kernel_parameters
    elif kernel_scheme == KernelScheme.custom:
        assert (
            "kernel_function" in kernel_kwargs
        ), "Custom kernel function must be specified."
        assert (
            "custom_kernel_function_scheme" in kernel_kwargs["kernel_function"]
        ), "Custom kernel function scheme must be specified."
        assert (
            "custom_kernel_function_kwargs" in kernel_kwargs["kernel_function"]
        ), "Custom kernel function kwargs must be specified."
        kernel_function, kernel_function_parameters = custom_kernel_function_resolver(
            custom_kernel_function_scheme=kernel_kwargs["kernel_function"][
                "custom_kernel_function_scheme"
            ],
            custom_kernel_function_kwargs=kernel_kwargs["kernel_function"][
                "custom_kernel_function_kwargs"
            ],
        )

        assert (
            "custom_kernel_function_input_shape" in kernel_kwargs["kernel_function"]
        ), "Input shape must be specified."
        kernel = CustomKernel(
            kernel_function=kernel_function,
            preprocess_function=lambda x: x.reshape(
                -1,
                *kernel_kwargs["kernel_function"]["custom_kernel_function_input_shape"],
            ),
        )
        kernel_parameters = kernel.generate_parameters(
            {"custom": kernel_function_parameters}
        )
        return kernel, kernel_parameters
    elif kernel_scheme in [
        KernelScheme.svgp,
        KernelScheme.diagonal_svgp,
        KernelScheme.log_svgp,
        KernelScheme.kernelised_svgp,
    ]:
        assert (
            "reference_kernel" in kernel_kwargs
        ), "Reference kernel must be specified."
        assert (
            "kernel_scheme" in kernel_kwargs["reference_kernel"]
        ), "Reference kernel kwargs must be specified."
        assert (
            "kernel_kwargs" in kernel_kwargs["reference_kernel"]
        ), "Reference kernel kwargs must be specified."
        assert (
            "kernel_parameters" in kernel_kwargs["reference_kernel"]
        ), "Reference kernel parameters must be specified."
        reference_kernel, reference_kernel_parameters = kernel_resolver(
            kernel_scheme=kernel_kwargs["reference_kernel"]["kernel_scheme"],
            kernel_kwargs=kernel_kwargs["reference_kernel"]["kernel_kwargs"],
            kernel_parameters=kernel_kwargs["reference_kernel"]["kernel_parameters"],
        )

        assert (
            "observation_noise" in kernel_kwargs
        ), "Observation noise must be specified."
        assert "inducing_points" in kernel_kwargs, "Inducing points must be specified."
        assert "training_points" in kernel_kwargs, "Training points must be specified."
        assert (
            "diagonal_regularisation" in kernel_kwargs
        ), "Diagonal regularisation must be specified."
        assert (
            "is_diagonal_regularisation_absolute_scale" in kernel_kwargs
        ), "Is diagonal regularisation absolute scale must be specified."
        if kernel_scheme == KernelScheme.svgp:
            kernel = SVGPKernel(
                reference_kernel=reference_kernel,
                reference_kernel_parameters=reference_kernel_parameters,
                log_observation_noise=jnp.log(kernel_kwargs["observation_noise"]),
                inducing_points=kernel_kwargs["inducing_points"],
                training_points=kernel_kwargs["training_points"],
                diagonal_regularisation=kernel_kwargs["diagonal_regularisation"],
                is_diagonal_regularisation_absolute_scale=kernel_kwargs[
                    "is_diagonal_regularisation_absolute_scale"
                ],
            )
            (
                el_matrix_lower_triangle,
                el_matrix_log_diagonal,
            ) = kernel.initialise_el_matrix_parameters()
            kernel_parameters = kernel.generate_parameters(
                {
                    "el_matrix_lower_triangle": el_matrix_lower_triangle,
                    "el_matrix_log_diagonal": el_matrix_log_diagonal,
                }
            )
            return kernel, kernel_parameters
        elif kernel_scheme == KernelScheme.diagonal_svgp:
            kernel = DiagonalSVGPKernel(
                reference_kernel=reference_kernel,
                reference_kernel_parameters=reference_kernel_parameters,
                log_observation_noise=jnp.log(kernel_kwargs["observation_noise"]),
                inducing_points=kernel_kwargs["inducing_points"],
                training_points=kernel_kwargs["training_points"],
                diagonal_regularisation=kernel_kwargs["diagonal_regularisation"],
                is_diagonal_regularisation_absolute_scale=kernel_kwargs[
                    "is_diagonal_regularisation_absolute_scale"
                ],
            )
            el_matrix_log_diagonal = kernel.initialise_diagonal_parameters()
            kernel_parameters = kernel.generate_parameters(
                {
                    "el_matrix_log_diagonal": el_matrix_log_diagonal,
                }
            )
            return kernel, kernel_parameters
        elif kernel_scheme == KernelScheme.log_svgp:
            kernel = LogSVGPKernel(
                reference_kernel=reference_kernel,
                reference_kernel_parameters=reference_kernel_parameters,
                log_observation_noise=jnp.log(kernel_kwargs["observation_noise"]),
                inducing_points=kernel_kwargs["inducing_points"],
                training_points=kernel_kwargs["training_points"],
                diagonal_regularisation=kernel_kwargs["diagonal_regularisation"],
                is_diagonal_regularisation_absolute_scale=kernel_kwargs[
                    "is_diagonal_regularisation_absolute_scale"
                ],
            )
            log_el_matrix = kernel.initialise_el_matrix_parameters()
            kernel_parameters = kernel.generate_parameters(
                {
                    "log_el_matrix": log_el_matrix,
                }
            )
            return kernel, kernel_parameters
        elif kernel_scheme == KernelScheme.kernelised_svgp:
            assert "base_kernel" in kernel_kwargs, "Base kernel must be specified."

            assert (
                "kernel_scheme" in kernel_kwargs["base_kernel"]
            ), "Base kernel kwargs must be specified."
            assert (
                "kernel_kwargs" in kernel_kwargs["base_kernel"]
            ), "Base kernel kwargs must be specified."
            assert (
                "kernel_parameters" in kernel_kwargs["base_kernel"]
            ), "Base kernel parameters must be specified."
            base_kernel, base_kernel_parameters = kernel_resolver(
                kernel_scheme=kernel_kwargs["base_kernel"]["kernel_scheme"],
                kernel_kwargs=kernel_kwargs["base_kernel"]["kernel_kwargs"],
                kernel_parameters=kernel_kwargs["base_kernel"]["kernel_parameters"],
            )
            kernel = KernelisedSVGPKernel(
                base_kernel=base_kernel,
                reference_kernel=reference_kernel,
                reference_kernel_parameters=reference_kernel_parameters,
                log_observation_noise=kernel_kwargs["log_observation_noise"],
                inducing_points=kernel_kwargs["inducing_points"],
                training_points=kernel_kwargs["training_points"],
                diagonal_regularisation=kernel_kwargs["diagonal_regularisation"],
                is_diagonal_regularisation_absolute_scale=kernel_kwargs[
                    "is_diagonal_regularisation_absolute_scale"
                ],
            )
            kernel_parameters = kernel.generate_parameters(
                {
                    "base_kernel": base_kernel_parameters,
                }
            )
            return kernel, kernel_parameters
    else:
        raise NotImplementedError(f"Kernel scheme {kernel_scheme} not implemented.")
