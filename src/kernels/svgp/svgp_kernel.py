from typing import Callable, Dict, Literal, Tuple, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_solve

from src.kernels.base import KernelBase, KernelBaseParameters
from src.kernels.svgp.base import SVGPBaseKernel, SVGPBaseKernelParameters
from src.utils.custom_types import JaxArrayType


class SVGPKernelParameters(SVGPBaseKernelParameters):
    """
    el_matrix_lower_triangle is a lower triangle of the L matrix
    el_matrix_log_diagonal is the logarithm of the diagonal of the L matrix
    combining them such that:
        L = el_matrix_lower_triangle + diagonalise(exp(el_matrix_log_diagonal))
    and
        sigma_matrix = L @ L.T
    """

    el_matrix_lower_triangle: JaxArrayType[Literal["float64"]]
    el_matrix_log_diagonal: JaxArrayType[Literal["float64"]]


class SVGPKernel(SVGPBaseKernel):
    """
    The stochastic variational Gaussian process kernel as defined in Titsias (2009).
    """

    Parameters = SVGPKernelParameters

    def __init__(
        self,
        reference_kernel: KernelBase,
        reference_kernel_parameters: KernelBaseParameters,
        log_observation_noise: float,
        inducing_points: jnp.ndarray,
        training_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        super().__init__(
            reference_kernel_parameters=reference_kernel_parameters,
            reference_kernel=reference_kernel,
            preprocess_function=preprocess_function,
            log_observation_noise=log_observation_noise,
            inducing_points=inducing_points,
            training_points=training_points,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict] = None
    ) -> SVGPKernelParameters:
        if parameters is None:
            (
                el_matrix_lower_triangle,
                el_matrix_log_diagonal,
            ) = self.initialise_el_matrix_parameters()
            return SVGPKernelParameters(
                el_matrix_lower_triangle=el_matrix_lower_triangle,
                el_matrix_log_diagonal=el_matrix_log_diagonal,
            )
        return SVGPKernelParameters(
            el_matrix_lower_triangle=parameters["el_matrix_lower_triangle"],
            el_matrix_log_diagonal=parameters["el_matrix_log_diagonal"],
        )

    def initialise_el_matrix_parameters(
        self,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialise the L matrix where:
            sigma_matrix = L @ L.T

        Returns:
            el_matrix_lower_triangle
            el_matrix_log_diagonal

        """
        reference_gaussian_measure_observation_precision = 1 / jnp.exp(
            self.log_observation_noise
        )
        cholesky_decomposition = jnp.linalg.cholesky(
            self.reference_gram_inducing
            + reference_gaussian_measure_observation_precision
            * self.gram_inducing_train
            @ self.gram_inducing_train.T
        )
        inverse_cholesky_decomposition = jnp.linalg.inv(cholesky_decomposition)
        el_matrix_lower_triangle = jnp.tril(inverse_cholesky_decomposition, k=-1)
        el_matrix_log_diagonal = jnp.log(
            jnp.clip(
                jnp.diag(inverse_cholesky_decomposition),
                self.diagonal_regularisation,
                None,
            )
        )
        return el_matrix_lower_triangle, el_matrix_log_diagonal

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, SVGPKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        reference_gram_x1_inducing = self.reference_kernel.calculate_gram(
            parameters=self.reference_kernel_parameters,
            x1=x1,
            x2=self.inducing_points,
        )

        reference_gram_x2_inducing = self.reference_kernel.calculate_gram(
            parameters=self.reference_kernel_parameters,
            x1=x2,
            x2=self.inducing_points,
        )

        reference_gram_x1_x2 = self.reference_kernel.calculate_gram(
            parameters=self.reference_kernel_parameters,
            x1=x1,
            x2=x2,
        )
        el_matrix_lower_triangle = jnp.tril(parameters.el_matrix_lower_triangle, k=-1)
        el_matrix = el_matrix_lower_triangle + jnp.diag(
            jnp.exp(parameters.el_matrix_log_diagonal),
        )
        sigma_matrix = el_matrix.T @ el_matrix
        return (
            reference_gram_x1_x2
            - (
                reference_gram_x1_inducing
                @ cho_solve(
                    c_and_lower=self.reference_gram_inducing_cholesky_decomposition_and_lower,
                    b=reference_gram_x2_inducing.T,
                )
            )
            + reference_gram_x1_inducing @ sigma_matrix @ reference_gram_x2_inducing.T
        )
