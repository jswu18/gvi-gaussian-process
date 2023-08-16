from typing import Dict, Tuple, Union

import jax.numpy as jnp
import orbax
import yaml
from flax.core.frozen_dict import FrozenDict

from experiments.shared.resolvers.nn_function import nn_function_resolver
from experiments.shared.resolvers.nngp_kernel_function import (
    nngp_kernel_function_resolver,
)
from experiments.shared.schemas import KernelSchema
from src.kernels import CustomKernel, CustomMappingKernel, MultiOutputKernel
from src.kernels.approximate import (
    CustomApproximateKernel,
    CustomMappingApproximateKernel,
    DiagonalSVGPKernel,
    KernelisedSVGPKernel,
    LogSVGPKernel,
    SVGPKernel,
)
from src.kernels.base import KernelBase, KernelBaseParameters
from src.kernels.non_stationary import PolynomialKernel
from src.kernels.non_stationary.base import NonStationaryKernelBase


def kernel_resolver(
    kernel_config: Union[FrozenDict, Dict],
) -> Tuple[KernelBase, KernelBaseParameters]:
    if "load_config_paths" in kernel_config:
        assert (
            "config_path" in kernel_config["load_config_paths"]
        ), "Must have kernel config path"
        assert (
            "parameters_path" in kernel_config["load_config_paths"]
        ), "Must have parameters path"
        with open(kernel_config["load_config_paths"]["config_path"], "r") as file:
            loaded_kernel_config = yaml.safe_load(file)

        kernel, _ = kernel_resolver(loaded_kernel_config["kernel"])
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(
            kernel_config["load_config_paths"]["parameters_path"]
        )
        kernel_parameters = kernel.Parameters.construct(**ckpt["kernel"])
        return kernel, kernel_parameters
    assert "kernel_schema" in kernel_config, "Kernel schema must be specified."
    assert "kernel_kwargs" in kernel_config, "Kernel kwargs must be specified."
    assert "kernel_parameters" in kernel_config, "Kernel parameters must be specified."
    kernel_schema: KernelSchema = kernel_config["kernel_schema"]
    kernel_kwargs: Union[FrozenDict, Dict] = kernel_config["kernel_kwargs"]
    kernel_parameters_config: Union[FrozenDict, Dict] = kernel_config[
        "kernel_parameters"
    ]

    if kernel_schema == KernelSchema.polynomial:
        assert (
            "polynomial_degree" in kernel_kwargs
        ), "Polynomial degree must be specified."
        kernel = PolynomialKernel(
            polynomial_degree=kernel_kwargs["polynomial_degree"],
        )

        assert (
            "log_constant" in kernel_parameters_config
        ), "Log constant must be specified."
        assert (
            "log_scaling" in kernel_parameters_config
        ), "Log scaling must be specified."
        kernel_parameters = kernel.generate_parameters(
            {
                "log_constant": kernel_parameters_config["log_constant"],
                "log_scaling": kernel_parameters_config["log_scaling"],
            }
        )
        return kernel, kernel_parameters
    elif kernel_schema == KernelSchema.multi_output:
        if "repeat" in kernel_kwargs:
            kernel_list = []
            kernel_parameters_list = []
            for _ in range(kernel_kwargs["repeat"]):
                kernel, kernel_parameters = kernel_resolver(
                    kernel_config=kernel_kwargs,
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
                kernel_kwargs, kernel_parameters_config
            ):
                kernel, kernel_parameters = kernel_resolver(
                    kernel_config=kernel_kwargs[kernel_kwargs_]
                )
                kernel_list.append(kernel)
                kernel_parameters_list.append(kernel_parameters)
            kernel = MultiOutputKernel(kernels=kernel_list)
            kernel_parameters = kernel.generate_parameters(
                {"kernels": kernel_parameters_list}
            )
            return kernel, kernel_parameters
    elif kernel_schema == KernelSchema.custom:
        assert (
            "nngp_kernel_function_kwargs" in kernel_kwargs
        ), "Custom kernel function kwargs must be specified."
        kernel_function, kernel_function_parameters = nngp_kernel_function_resolver(
            nngp_kernel_function_kwargs=kernel_kwargs["nngp_kernel_function_kwargs"],
        )

        assert (
            "input_shape" in kernel_kwargs["nngp_kernel_function_kwargs"]
        ), "Input shape must be specified."
        kernel = CustomKernel(
            kernel_function=kernel_function,
            preprocess_function=lambda x: x.reshape(
                -1,
                *kernel_kwargs["nngp_kernel_function_kwargs"]["input_shape"],
            ),
        )
        kernel_parameters = kernel.generate_parameters(
            {"custom": kernel_function_parameters}
        )
        return kernel, kernel_parameters
    elif kernel_schema == KernelSchema.custom_mapping:
        assert "base_kernel" in kernel_kwargs, "Base kernel must be specified."
        assert (
            "nn_function_kwargs" in kernel_kwargs
        ), "Feature mapping must be specified."

        base_kernel, base_kernel_parameters = kernel_resolver(
            kernel_config=kernel_kwargs["base_kernel"],
        )
        assert isinstance(
            base_kernel, NonStationaryKernelBase
        ), "Base kernel must be non-stationary."
        feature_mapping, feature_mapping_parameters = nn_function_resolver(
            nn_function_kwargs=kernel_kwargs["nn_function_kwargs"],
        )
        kernel = CustomMappingKernel(
            base_kernel=base_kernel,
            feature_mapping=feature_mapping,
        )
        kernel_parameters = kernel.generate_parameters(
            {
                "base_kernel": base_kernel_parameters,
                "feature_mapping": feature_mapping_parameters,
            }
        )
        return kernel, kernel_parameters
    elif kernel_schema == KernelSchema.custom_approximate:
        assert (
            "nngp_kernel_function_kwargs" in kernel_kwargs
        ), "Custom kernel function kwargs must be specified."
        assert "inducing_points" in kernel_kwargs, "Inducing points must be specified."
        assert (
            "diagonal_regularisation" in kernel_kwargs
        ), "Diagonal regularisation must be specified."
        assert (
            "is_diagonal_regularisation_absolute_scale" in kernel_kwargs
        ), "Is diagonal regularisation absolute scale must be specified."
        assert (
            "input_shape" in kernel_kwargs["nngp_kernel_function_kwargs"]
        ), "Input shape must be specified."
        kernel_function, kernel_function_parameters = nngp_kernel_function_resolver(
            nngp_kernel_function_kwargs=kernel_kwargs["nngp_kernel_function_kwargs"],
        )
        kernel = CustomApproximateKernel(
            kernel_function=kernel_function,
            inducing_points=kernel_kwargs["inducing_points"],
            diagonal_regularisation=kernel_kwargs["diagonal_regularisation"],
            is_diagonal_regularisation_absolute_scale=kernel_kwargs[
                "is_diagonal_regularisation_absolute_scale"
            ],
            preprocess_function=lambda x: x.reshape(
                -1,
                *kernel_kwargs["nngp_kernel_function_kwargs"]["input_shape"],
            ),
        )
        kernel_parameters = kernel.generate_parameters(
            {"custom": kernel_function_parameters}
        )
        return kernel, kernel_parameters
    elif kernel_schema == KernelSchema.custom_mapping_approximate:
        assert "base_kernel" in kernel_kwargs, "Base kernel must be specified."
        assert (
            "nn_function_kwargs" in kernel_kwargs
        ), "Feature mapping must be specified."
        assert "inducing_points" in kernel_kwargs, "Inducing points must be specified."
        assert (
            "diagonal_regularisation" in kernel_kwargs
        ), "Diagonal regularisation must be specified."
        assert (
            "is_diagonal_regularisation_absolute_scale" in kernel_kwargs
        ), "Is diagonal regularisation absolute scale must be specified."

        base_kernel, base_kernel_parameters = kernel_resolver(
            kernel_config=kernel_kwargs["base_kernel"],
        )
        assert isinstance(
            base_kernel, NonStationaryKernelBase
        ), "Base kernel must be non-stationary."
        feature_mapping, feature_mapping_parameters = nn_function_resolver(
            nn_function_kwargs=kernel_kwargs["nn_function_kwargs"],
        )
        kernel = CustomMappingApproximateKernel(
            base_kernel=base_kernel,
            feature_mapping=feature_mapping,
            inducing_points=kernel_kwargs["inducing_points"],
            diagonal_regularisation=kernel_kwargs["diagonal_regularisation"],
            is_diagonal_regularisation_absolute_scale=kernel_kwargs[
                "is_diagonal_regularisation_absolute_scale"
            ],
        )
        kernel_parameters = kernel.generate_parameters(
            {
                "base_kernel": base_kernel_parameters,
                "feature_mapping": feature_mapping_parameters,
            }
        )
        return kernel, kernel_parameters
    elif kernel_schema in [
        KernelSchema.svgp,
        KernelSchema.diagonal_svgp,
        KernelSchema.log_svgp,
        KernelSchema.kernelised_svgp,
    ]:
        assert (
            "reference_kernel" in kernel_kwargs
        ), "Reference kernel must be specified."
        reference_kernel, reference_kernel_parameters = kernel_resolver(
            kernel_config=kernel_kwargs["reference_kernel"],
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
        if kernel_schema == KernelSchema.svgp:
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
        elif kernel_schema == KernelSchema.diagonal_svgp:
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
        elif kernel_schema == KernelSchema.log_svgp:
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
        elif kernel_schema == KernelSchema.kernelised_svgp:
            assert "base_kernel" in kernel_kwargs, "Base kernel must be specified."
            base_kernel, base_kernel_parameters = kernel_resolver(
                kernel_config=kernel_kwargs["base_kernel"],
            )
            kernel = KernelisedSVGPKernel(
                base_kernel=base_kernel,
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
            kernel_parameters = kernel.generate_parameters(
                {
                    "base_kernel": base_kernel_parameters,
                }
            )
            return kernel, kernel_parameters
    else:
        raise NotImplementedError(f"Kernel schema {kernel_schema} not implemented.")
