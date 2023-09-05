from typing import Callable, Dict, Literal, Tuple, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_solve

from src.kernels.approximate.svgp.base import SVGPBaseKernel, SVGPBaseKernelParameters
from src.kernels.base import KernelBase, KernelBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG
from src.utils.custom_types import JaxArrayType


class DiagonalSVGPKernelParameters(SVGPBaseKernelParameters):
    """
    The diagonal parameters used to parameterise the SVGP kernel.
    """

    log_el_matrix_diagonal: JaxArrayType[Literal["float64"]]


class DiagonalSVGPKernel(SVGPBaseKernel):
    """
    A diagonal parameterisation of the SVGP kernel.
    """

    Parameters = DiagonalSVGPKernelParameters

    def __init__(
        self,
        regulariser_kernel: KernelBase,
        regulariser_kernel_parameters: KernelBaseParameters,
        log_observation_noise: float,
        inducing_points: jnp.ndarray,
        training_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        super().__init__(
            regulariser_kernel_parameters=regulariser_kernel_parameters,
            regulariser_kernel=regulariser_kernel,
            preprocess_function=preprocess_function,
            log_observation_noise=log_observation_noise,
            inducing_points=inducing_points,
            training_points=training_points,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
        )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict] = None
    ) -> DiagonalSVGPKernelParameters:
        if parameters is None:
            return DiagonalSVGPKernelParameters(
                log_el_matrix_diagonal=self.initialise_diagonal_parameters()
            )
        return DiagonalSVGPKernelParameters(
            log_el_matrix_diagonal=parameters["log_el_matrix_diagonal"]
        )

    def initialise_diagonal_parameters(
        self,
    ) -> jnp.ndarray:
        """
        Initialise the diagonal parameters of the kernel by
        taking the diagonal of the Cholesky decomposition.
        """
        regulariser_gaussian_measure_observation_precision = 1 / jnp.exp(
            self.log_observation_noise
        )
        cholesky_decomposition = jnp.linalg.cholesky(
            self.regulariser_gram_inducing
            + regulariser_gaussian_measure_observation_precision
            * self.gram_inducing_train
            @ self.gram_inducing_train.T
        )
        inv_cholesky_decomposition = jnp.linalg.inv(cholesky_decomposition)
        return jnp.diag(
            jnp.log(
                jnp.clip(
                    inv_cholesky_decomposition @ inv_cholesky_decomposition.T,
                    self.diagonal_regularisation,
                    None,
                )
            )
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, DiagonalSVGPKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        regulariser_gram_x1_inducing = self.regulariser_kernel.calculate_gram(
            parameters=self.regulariser_kernel_parameters,
            x1=x1,
            x2=self.inducing_points,
        )

        regulariser_gram_x2_inducing = self.regulariser_kernel.calculate_gram(
            parameters=self.regulariser_kernel_parameters,
            x1=x2,
            x2=self.inducing_points,
        )

        regulariser_gram_x1_x2 = self.regulariser_kernel.calculate_gram(
            parameters=self.regulariser_kernel_parameters,
            x1=x1,
            x2=x2,
        )

        sigma_diagonal = jnp.exp(parameters.log_el_matrix_diagonal)
        return (
            regulariser_gram_x1_x2
            - (
                regulariser_gram_x1_inducing
                @ cho_solve(
                    c_and_lower=self.regulariser_gram_inducing_cholesky_decomposition_and_lower,
                    b=regulariser_gram_x2_inducing.T,
                )
            )
            + regulariser_gram_x1_inducing
            @ jnp.multiply(sigma_diagonal[:, None], regulariser_gram_x2_inducing.T)
        )
