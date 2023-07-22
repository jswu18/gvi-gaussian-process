from typing import Any, Callable, Dict, Literal, Tuple, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.approximate.base import (
    ApproximateBaseKernel,
    ApproximateBaseKernelParameters,
)
from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import JaxArrayType
from src.utils.matrix_operations import add_diagonal_regulariser

PRNGKey = Any  # pylint: disable=invalid-name


class StochasticVariationalKernelParameters(ApproximateBaseKernelParameters):
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


class StochasticVariationalKernel(ApproximateBaseKernel):
    """
    The stochastic variational Gaussian process kernel as defined in Titsias (2009).
    """

    Parameters = StochasticVariationalKernelParameters

    def __init__(
        self,
        reference_kernel: KernelBase,
        reference_kernel_parameters: KernelBaseParameters,
        log_observation_noise: float,
        inducing_points: jnp.ndarray,
        training_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        el_matrix_diagonal_lower_bound: float = 1e-3,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        """
        Defining the stochastic variational Gaussian process kernel using the reference Gaussian measure
        and inducing points.

        Args:
            reference_kernel_parameters: the parameters of the reference kernel.
            log_observation_noise: the log observation noise of the model
            reference_kernel: the kernel of the reference Gaussian measure.
            inducing_points: the inducing points of the stochastic variational Gaussian process.
            training_points: the training points of the stochastic variational Gaussian process.
            diagonal_regularisation: the diagonal regularisation used to stabilise the Cholesky decomposition.
            el_matrix_diagonal_lower_bound: lower bound (clip) the diagonals of the L parameter matrix for Sigma = LL^T
            is_diagonal_regularisation_absolute_scale: whether the diagonal regularisation is an absolute scale.
        """
        self.log_observation_noise = log_observation_noise
        self.number_of_dimensions = inducing_points.shape[1]
        self.inducing_points = inducing_points
        self.training_points = training_points
        self.diagonal_regularisation = diagonal_regularisation
        self.el_matrix_diagonal_lower_bound = el_matrix_diagonal_lower_bound
        self.is_diagonal_regularisation_absolute_scale = (
            is_diagonal_regularisation_absolute_scale
        )
        super().__init__(
            reference_kernel_parameters=reference_kernel_parameters,
            reference_kernel=reference_kernel,
            preprocess_function=preprocess_function,
        )
        self.reference_gram_inducing = self.reference_kernel.calculate_gram(
            parameters=reference_kernel_parameters,
            x1=inducing_points,
            x2=inducing_points,
        )
        self.reference_gram_inducing_cholesky_decomposition_and_lower = cho_factor(
            add_diagonal_regulariser(
                matrix=self.reference_gram_inducing,
                diagonal_regularisation=diagonal_regularisation,
                is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
            )
        )
        self.gram_inducing_train = self.reference_kernel.calculate_gram(
            parameters=reference_kernel_parameters,
            x1=inducing_points,
            x2=training_points,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(self, parameters: Union[FrozenDict, Dict]) -> Parameters:
        """
        Generates a Pydantic model of the parameters for Stochastic Variational Gaussian Process Kernels.

        Args:
            parameters: A dictionary of the parameters for Stochastic Variational Gaussian Process Kernels.

        Returns: A Pydantic model of the parameters for Stochastic Variational Gaussian Process Kernels.

        """
        return StochasticVariationalKernel.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey = None,
    ) -> StochasticVariationalKernelParameters:
        """
        Initialise each parameter of the Stochastic Variational Gaussian Process Kernel with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters. Not required in this case becasue
                the parameters are initialised deterministically.

        Returns: A Pydantic model of the parameters for Stochastic Variational Gaussian Process Kernels.

        """
        # raise warning if key is not None
        (
            el_matrix_lower_triangle,
            el_matrix_log_diagonal,
        ) = self.initialise_el_matrix_parameters()
        return StochasticVariationalKernel.Parameters(
            el_matrix_lower_triangle=el_matrix_lower_triangle,
            el_matrix_log_diagonal=el_matrix_log_diagonal,
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
                self.el_matrix_diagonal_lower_bound,
                None,
            )
        )
        return el_matrix_lower_triangle, el_matrix_log_diagonal

    def _calculate_gram(
        self,
        parameters: StochasticVariationalKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        reference_gram_x1_inducing = self.reference_kernel.calculate_gram(
            x=x,
            y=self.inducing_points,
        )

        # if y is None, compute for x and x
        if jnp.array_equal(x1, x2):
            reference_gram_x2_inducing = reference_gram_x1_inducing
        else:
            reference_gram_x2_inducing = self.reference_kernel.calculate_gram(
                x1=x1,
                x2=self.inducing_points,
            )

        reference_gram_x1_x2 = self.reference_kernel.calculate_gram(x1=x1, x2=x2)
        el_matrix_lower_triangle = jnp.tril(parameters.el_matrix_lower_triangle, k=-1)
        el_matrix = el_matrix_lower_triangle + jnp.diag(
            jnp.clip(
                jnp.exp(parameters.el_matrix_log_diagonal),
                self.el_matrix_diagonal_lower_bound,
                None,
            )
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