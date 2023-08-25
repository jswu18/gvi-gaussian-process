from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
import orbax
import yaml
from flax.core.frozen_dict import FrozenDict

from experiments.shared.resolvers.nn_function import nn_function_resolver
from experiments.shared.resolvers.nngp_kernel_function import (
    nngp_kernel_function_resolver,
)
from experiments.shared.schemas import KernelSchema
from src.kernels import (
    CustomKernel,
    CustomKernelParameters,
    CustomMappingKernel,
    CustomMappingKernelParameters,
    MultiOutputKernel,
)
from src.kernels.approximate import (
    DecomposedSVGPKernel,
    DiagonalSVGPKernel,
    FixedSparsePosteriorKernel,
    FixedSparsePosteriorKernelParameters,
    KernelisedSVGPKernel,
    LogSVGPKernel,
    SparsePosteriorKernel,
    SparsePosteriorKernelParameters,
)
from src.kernels.approximate.extended_svgp.base import (
    ExtendedSVGPBaseKernel,
    ExtendedSVGPBaseKernelParameters,
)
from src.kernels.approximate.extended_svgp.decomposed_svgp_kernel import (
    DecomposedSVGPKernelParameters,
)
from src.kernels.approximate.extended_svgp.diagonal_svgp_kernel import (
    DiagonalSVGPKernelParameters,
)
from src.kernels.approximate.extended_svgp.kernelised_svgp_kernel import (
    KernelisedSVGPKernelParameters,
)
from src.kernels.approximate.extended_svgp.log_svgp_kernel import (
    LogSVGPKernelParameters,
)
from src.kernels.base import KernelBase, KernelBaseParameters
from src.kernels.non_stationary import (
    InnerProductKernel,
    InnerProductKernelParameters,
    PolynomialKernel,
    PolynomialKernelParameters,
)
from src.kernels.non_stationary.base import NonStationaryKernelBase


def resolve_existing_kernel(
    config_path: str,
    parameter_path: str,
) -> Tuple[KernelBase, KernelBaseParameters]:
    with open(config_path, "r") as file:
        loaded_kernel_config = yaml.safe_load(file)

    kernel, _ = kernel_resolver(loaded_kernel_config["kernel"])
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(parameter_path)
    kernel_parameters = kernel.Parameters.construct(**ckpt["kernel"])
    return kernel, kernel_parameters


def _resolve_inner_product_kernel(
    kernel_parameters_config: Union[FrozenDict, Dict],
) -> Tuple[InnerProductKernel, InnerProductKernelParameters]:
    assert "scaling" in kernel_parameters_config, "Scaling must be specified."
    kernel = InnerProductKernel()
    kernel_parameters = kernel.generate_parameters(
        {
            "log_scaling": jnp.log(kernel_parameters_config["scaling"]),
        }
    )
    return kernel, kernel_parameters


def _resolve_polynomial_kernel(
    kernel_kwargs_config: Union[FrozenDict, Dict],
    kernel_parameters_config: Union[FrozenDict, Dict],
) -> Tuple[PolynomialKernel, PolynomialKernelParameters]:
    assert (
        "polynomial_degree" in kernel_kwargs_config
    ), "Polynomial degree must be specified."
    kernel = PolynomialKernel(
        polynomial_degree=kernel_kwargs_config["polynomial_degree"],
    )

    assert "constant" in kernel_parameters_config, "Constant must be specified."
    assert "scaling" in kernel_parameters_config, "Scaling must be specified."
    kernel_parameters = kernel.generate_parameters(
        {
            "log_constant": jnp.log(kernel_parameters_config["constant"]),
            "log_scaling": jnp.log(kernel_parameters_config["scaling"]),
        }
    )
    return kernel, kernel_parameters


def _resolve_nngp_kernel(
    kernel_kwargs_config: Union[FrozenDict, Dict],
) -> Tuple[CustomKernel, CustomKernelParameters]:
    kernel_function, kernel_function_parameters = nngp_kernel_function_resolver(
        nngp_kernel_function_kwargs=kernel_kwargs_config,
    )
    assert "input_shape" in kernel_kwargs_config, "Input shape must be specified."
    kernel = CustomKernel(
        kernel_function=kernel_function,
        preprocess_function=lambda x: x.reshape(
            -1,
            *kernel_kwargs_config["input_shape"],
        ),
    )
    kernel_parameters = kernel.generate_parameters(
        {"custom": kernel_function_parameters}
    )
    return kernel, kernel_parameters


def _resolve_custom_mapping_kernel(
    kernel_kwargs_config: Union[FrozenDict, Dict],
) -> Tuple[CustomMappingKernel, CustomMappingKernelParameters]:
    assert "base_kernel" in kernel_kwargs_config, "Base kernel must be specified."
    assert (
        "nn_function_kwargs" in kernel_kwargs_config
    ), "Feature mapping must be specified."

    base_kernel, base_kernel_parameters = kernel_resolver(
        kernel_config=kernel_kwargs_config["base_kernel"],
    )
    assert isinstance(
        base_kernel, NonStationaryKernelBase
    ), "Base kernel must be non-stationary."
    feature_mapping, feature_mapping_parameters = nn_function_resolver(
        nn_function_kwargs=kernel_kwargs_config["nn_function_kwargs"],
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


def _resolve_sparse_posterior_kernel(
    kernel_kwargs_config: Union[FrozenDict, Dict],
    reference_kernel: Optional[KernelBase] = None,
    reference_kernel_parameters: Optional[KernelBaseParameters] = None,
) -> Tuple[SparsePosteriorKernel, SparsePosteriorKernelParameters]:
    assert (
        "inducing_points" in kernel_kwargs_config
    ), "Inducing points must be specified."
    assert (
        "diagonal_regularisation" in kernel_kwargs_config
    ), "Diagonal regularisation must be specified."
    assert (
        "is_diagonal_regularisation_absolute_scale" in kernel_kwargs_config
    ), "Is diagonal regularisation absolute scale must be specified."
    if not reference_kernel and not reference_kernel_parameters:
        assert "base_kernel" in kernel_kwargs_config, "Base kernel must be specified."
        base_kernel, base_kernel_parameters = kernel_resolver(
            kernel_config=kernel_kwargs_config["base_kernel"],
        )
    else:
        base_kernel = reference_kernel
        base_kernel_parameters = reference_kernel_parameters
    kernel = SparsePosteriorKernel(
        base_kernel=base_kernel,
        inducing_points=kernel_kwargs_config["inducing_points"],
        diagonal_regularisation=kernel_kwargs_config["diagonal_regularisation"],
        is_diagonal_regularisation_absolute_scale=kernel_kwargs_config[
            "is_diagonal_regularisation_absolute_scale"
        ],
    )
    kernel_parameters = kernel.generate_parameters(
        {"base_kernel": base_kernel_parameters}
    )
    return kernel, kernel_parameters


def _resolve_fixed_sparse_posterior_kernel(
    kernel_kwargs_config: Union[FrozenDict, Dict],
    reference_kernel: Optional[KernelBase] = None,
    reference_kernel_parameters: Optional[KernelBaseParameters] = None,
) -> Tuple[FixedSparsePosteriorKernel, FixedSparsePosteriorKernelParameters]:
    assert (
        "inducing_points" in kernel_kwargs_config
    ), "Inducing points must be specified."
    assert (
        "diagonal_regularisation" in kernel_kwargs_config
    ), "Diagonal regularisation must be specified."
    assert (
        "is_diagonal_regularisation_absolute_scale" in kernel_kwargs_config
    ), "Is diagonal regularisation absolute scale must be specified."
    if not reference_kernel and not reference_kernel_parameters:
        assert (
            "reference_kernel" in kernel_kwargs_config
        ), "Reference kernel must be specified."
        reference_kernel, reference_kernel_parameters = kernel_resolver(
            kernel_config=kernel_kwargs_config["base_kernel"],
        )
    if "base_kernel" not in kernel_kwargs_config:
        base_kernel = reference_kernel
        base_kernel_parameters = reference_kernel_parameters
    else:
        assert "base_kernel" in kernel_kwargs_config, "Base kernel must be specified."
        base_kernel, base_kernel_parameters = kernel_resolver(
            kernel_config=kernel_kwargs_config["base_kernel"],
        )

    kernel = FixedSparsePosteriorKernel(
        base_kernel=base_kernel,
        reference_kernel=reference_kernel,
        reference_kernel_parameters=reference_kernel_parameters,
        inducing_points=kernel_kwargs_config["inducing_points"],
        diagonal_regularisation=kernel_kwargs_config["diagonal_regularisation"],
        is_diagonal_regularisation_absolute_scale=kernel_kwargs_config[
            "is_diagonal_regularisation_absolute_scale"
        ],
    )
    kernel_parameters = kernel.generate_parameters(
        {"base_kernel": base_kernel_parameters}
    )
    return kernel, kernel_parameters


def _resolve_decomposed_svgp_kernel(
    kernel_kwargs_config: Union[FrozenDict, Dict],
    reference_kernel: KernelBase,
    reference_kernel_parameters: KernelBaseParameters,
) -> Tuple[DecomposedSVGPKernel, DecomposedSVGPKernelParameters]:
    kernel = DecomposedSVGPKernel(
        reference_kernel=reference_kernel,
        reference_kernel_parameters=reference_kernel_parameters,
        log_observation_noise=jnp.log(kernel_kwargs_config["observation_noise"]),
        inducing_points=kernel_kwargs_config["inducing_points"],
        training_points=kernel_kwargs_config["training_points"],
        diagonal_regularisation=kernel_kwargs_config["diagonal_regularisation"],
        is_diagonal_regularisation_absolute_scale=kernel_kwargs_config[
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


def _resolve_diagonal_svgp_kernel(
    kernel_kwargs_config: Union[FrozenDict, Dict],
    reference_kernel: KernelBase,
    reference_kernel_parameters: KernelBaseParameters,
) -> Tuple[DiagonalSVGPKernel, DiagonalSVGPKernelParameters]:
    kernel = DiagonalSVGPKernel(
        reference_kernel=reference_kernel,
        reference_kernel_parameters=reference_kernel_parameters,
        log_observation_noise=jnp.log(kernel_kwargs_config["observation_noise"]),
        inducing_points=kernel_kwargs_config["inducing_points"],
        training_points=kernel_kwargs_config["training_points"],
        diagonal_regularisation=kernel_kwargs_config["diagonal_regularisation"],
        is_diagonal_regularisation_absolute_scale=kernel_kwargs_config[
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


def _resolve_log_svgp_kernel(
    kernel_kwargs_config: Union[FrozenDict, Dict],
    reference_kernel: KernelBase,
    reference_kernel_parameters: KernelBaseParameters,
) -> Tuple[LogSVGPKernel, LogSVGPKernelParameters]:
    kernel = LogSVGPKernel(
        reference_kernel=reference_kernel,
        reference_kernel_parameters=reference_kernel_parameters,
        log_observation_noise=jnp.log(kernel_kwargs_config["observation_noise"]),
        inducing_points=kernel_kwargs_config["inducing_points"],
        training_points=kernel_kwargs_config["training_points"],
        diagonal_regularisation=kernel_kwargs_config["diagonal_regularisation"],
        is_diagonal_regularisation_absolute_scale=kernel_kwargs_config[
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


def _resolve_kernelised_svgp_kernel(
    kernel_kwargs_config: Union[FrozenDict, Dict],
    reference_kernel: KernelBase,
    reference_kernel_parameters: KernelBaseParameters,
) -> Tuple[KernelisedSVGPKernel, KernelisedSVGPKernelParameters]:
    if "base_kernel" not in kernel_kwargs_config:
        base_kernel = reference_kernel
        base_kernel_parameters = reference_kernel_parameters
    else:
        base_kernel, base_kernel_parameters = kernel_resolver(
            kernel_config=kernel_kwargs_config["base_kernel"],
        )
    kernel = KernelisedSVGPKernel(
        base_kernel=base_kernel,
        reference_kernel=reference_kernel,
        reference_kernel_parameters=reference_kernel_parameters,
        log_observation_noise=jnp.log(kernel_kwargs_config["observation_noise"]),
        inducing_points=kernel_kwargs_config["inducing_points"],
        training_points=kernel_kwargs_config["training_points"],
        diagonal_regularisation=kernel_kwargs_config["diagonal_regularisation"],
        is_diagonal_regularisation_absolute_scale=kernel_kwargs_config[
            "is_diagonal_regularisation_absolute_scale"
        ],
    )
    kernel_parameters = kernel.generate_parameters(
        {
            "base_kernel": base_kernel_parameters,
        }
    )
    return kernel, kernel_parameters


def _resolve_extended_svgp_kernel(
    kernel_schema: KernelSchema,
    kernel_kwargs_config: Union[FrozenDict, Dict],
    reference_kernel: Optional[KernelBase],
    reference_kernel_parameters: Optional[KernelBaseParameters],
) -> Tuple[ExtendedSVGPBaseKernel, ExtendedSVGPBaseKernelParameters]:
    if not reference_kernel and not reference_kernel_parameters:
        assert (
            "reference_kernel" in kernel_kwargs_config
        ), "Reference kernel must be specified."
        reference_kernel, reference_kernel_parameters = kernel_resolver(
            kernel_config=kernel_kwargs_config["reference_kernel"],
        )
    assert (
        "observation_noise" in kernel_kwargs_config
    ), "Observation noise must be specified."
    assert (
        "inducing_points" in kernel_kwargs_config
    ), "Inducing points must be specified."
    assert (
        "training_points" in kernel_kwargs_config
    ), "Training points must be specified."
    assert (
        "diagonal_regularisation" in kernel_kwargs_config
    ), "Diagonal regularisation must be specified."
    assert (
        "is_diagonal_regularisation_absolute_scale" in kernel_kwargs_config
    ), "Is diagonal regularisation absolute scale must be specified."
    if kernel_schema == KernelSchema.decomposed_svgp:
        return _resolve_decomposed_svgp_kernel(
            kernel_kwargs_config=kernel_kwargs_config,
            reference_kernel=reference_kernel,
            reference_kernel_parameters=reference_kernel_parameters,
        )
    elif kernel_schema == KernelSchema.diagonal_svgp:
        return _resolve_diagonal_svgp_kernel(
            kernel_kwargs_config=kernel_kwargs_config,
            reference_kernel=reference_kernel,
            reference_kernel_parameters=reference_kernel_parameters,
        )
    elif kernel_schema == KernelSchema.log_svgp:
        return _resolve_log_svgp_kernel(
            kernel_kwargs_config=kernel_kwargs_config,
            reference_kernel=reference_kernel,
            reference_kernel_parameters=reference_kernel_parameters,
        )
    elif kernel_schema == KernelSchema.kernelised_svgp:
        return _resolve_kernelised_svgp_kernel(
            kernel_kwargs_config=kernel_kwargs_config,
            reference_kernel=reference_kernel,
            reference_kernel_parameters=reference_kernel_parameters,
        )
    else:
        raise NotImplementedError(f"Kernel schema {kernel_schema} is not implemented.")


def kernel_resolver(
    kernel_config: Union[FrozenDict, Dict],
    reference_kernel: Optional[KernelBase] = None,
    reference_kernel_parameters: Optional[KernelBaseParameters] = None,
) -> Tuple[KernelBase, KernelBaseParameters]:
    assert "kernel_schema" in kernel_config, "Kernel schema must be specified."
    assert "kernel_kwargs" in kernel_config, "Kernel kwargs must be specified."
    assert "kernel_parameters" in kernel_config, "Kernel parameters must be specified."
    kernel_schema: KernelSchema = kernel_config["kernel_schema"]
    kernel_kwargs_config: Union[FrozenDict, Dict] = kernel_config["kernel_kwargs"]
    kernel_parameters_config: Union[FrozenDict, Dict] = kernel_config[
        "kernel_parameters"
    ]
    if kernel_schema == KernelSchema.inner_product:
        return _resolve_inner_product_kernel(
            kernel_parameters_config=kernel_parameters_config
        )
    elif kernel_schema == KernelSchema.polynomial:
        return _resolve_polynomial_kernel(
            kernel_kwargs_config=kernel_kwargs_config,
            kernel_parameters_config=kernel_parameters_config,
        )
    elif kernel_schema == KernelSchema.nngp:
        return _resolve_nngp_kernel(
            kernel_kwargs_config=kernel_kwargs_config,
        )
    elif kernel_schema == KernelSchema.custom_mapping:
        return _resolve_custom_mapping_kernel(
            kernel_kwargs_config=kernel_kwargs_config,
        )
    elif kernel_schema == KernelSchema.sparse_posterior:
        return _resolve_sparse_posterior_kernel(
            kernel_kwargs_config=kernel_kwargs_config,
            reference_kernel=reference_kernel,
            reference_kernel_parameters=reference_kernel_parameters,
        )
    elif kernel_schema == KernelSchema.fixed_sparse_posterior:
        return _resolve_fixed_sparse_posterior_kernel(
            kernel_kwargs_config=kernel_kwargs_config,
            reference_kernel=reference_kernel,
            reference_kernel_parameters=reference_kernel_parameters,
        )
    elif kernel_schema in [
        KernelSchema.decomposed_svgp,
        KernelSchema.diagonal_svgp,
        KernelSchema.log_svgp,
        KernelSchema.kernelised_svgp,
    ]:
        return _resolve_extended_svgp_kernel(
            kernel_schema=kernel_schema,
            kernel_kwargs_config=kernel_kwargs_config,
            reference_kernel=reference_kernel,
            reference_kernel_parameters=reference_kernel_parameters,
        )
    else:
        raise NotImplementedError(f"Kernel schema {kernel_schema} not implemented.")

    #
    # elif kernel_schema == KernelSchema.multi_output:
    #     if "repeat" in kernel_kwargs:
    #         kernel_list = []
    #         kernel_parameters_list = []
    #         for _ in range(kernel_kwargs["repeat"]):
    #             kernel, kernel_parameters = kernel_resolver(
    #                 kernel_config=kernel_kwargs,
    #             )
    #             kernel_list.append(kernel)
    #             kernel_parameters_list.append(kernel_parameters)
    #         kernel = MultiOutputKernel(kernels=kernel_list)
    #         kernel_parameters = kernel.generate_parameters(
    #             {"kernels": kernel_parameters_list}
    #         )
    #         return kernel, kernel_parameters
    #     else:
    #         kernel_list = []
    #         kernel_parameters_list = []
    #         for kernel_kwargs_, kernel_parameters_ in zip(
    #             kernel_kwargs, kernel_parameters_config
    #         ):
    #             kernel, kernel_parameters = kernel_resolver(
    #                 kernel_config=kernel_kwargs[kernel_kwargs_]
    #             )
    #             kernel_list.append(kernel)
    #             kernel_parameters_list.append(kernel_parameters)
    #         kernel = MultiOutputKernel(kernels=kernel_list)
    #         kernel_parameters = kernel.generate_parameters(
    #             {"kernels": kernel_parameters_list}
    #         )
    #         return kernel, kernel_parameters
