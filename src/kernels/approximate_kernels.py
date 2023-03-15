from abc import ABC
from typing import Any, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.reference_kernels import Kernel
from src.module import Module
from src.parameters.gaussian_measures.reference_gaussian_measure import (
    ReferenceGaussianMeasureParameters,
)
from src.parameters.kernels.approximate_kernels import (
    ApproximateKernelParameters,
    StochasticVariationalGaussianProcessKernelParameters,
)
from src.utils.matrix_operations import add_diagonal_regulariser

PRNGKey = Any  # pylint: disable=invalid-name


class ApproximateKernel(Kernel, ABC):
    """
    Approximate kernels which are defined with respect to a reference Gaussian measure.
    """

    Parameters = ApproximateKernelParameters

    def __init__(
        self,
        reference_gaussian_measure_parameters: ReferenceGaussianMeasureParameters,
        reference_kernel: Kernel,
    ):
        """
        Defining the kernel with respect to a reference Gaussian measure.

        Args:
            reference_gaussian_measure_parameters: the parameters of the reference Gaussian measure.
            reference_kernel: the kernel of the reference Gaussian measure.
        """

        self.reference_gaussian_measure_parameters = (
            reference_gaussian_measure_parameters
        )

        # define a jit-compiled version of the reference kernel gram matrix using the reference kernel parameters
        self.calculate_reference_gram = jit(
            lambda x, y=None: reference_kernel.calculate_gram(
                parameters=reference_gaussian_measure_parameters.kernel,
                x=x,
                y=y,
            )
        )


class StochasticVariationalGaussianProcessKernel(ApproximateKernel):
    """
    The stochastic variational Gaussian process kernel as defined in Titsias (2009).
    """

    Parameters = StochasticVariationalGaussianProcessKernelParameters

    def __init__(
        self,
        reference_gaussian_measure_parameters: ReferenceGaussianMeasureParameters,
        reference_kernel: Kernel,
        inducing_points: jnp.ndarray,
        training_points: jnp.ndarray,
        log_shift: float = 1e-5,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
    ):
        """
        Defining the stochastic variational Gaussian process kernel using the reference Gaussian measure
        and inducing points.

        Args:
            reference_gaussian_measure_parameters: the parameters of the reference Gaussian measure.
            reference_kernel: the kernel of the reference Gaussian measure.
            inducing_points: the inducing points of the stochastic variational Gaussian process.
            training_points: the training points of the stochastic variational Gaussian process.
            log_shift: the shift applied before logarithm of el_matrix and after exponentiation of log_el_matrix
            diagonal_regularisation: the diagonal regularisation used to stabilise the Cholesky decomposition.
            is_diagonal_regularisation_absolute_scale: whether the diagonal regularisation is an absolute scale.
        """
        super().__init__(
            reference_gaussian_measure_parameters,
            reference_kernel,
        )
        self.number_of_dimensions = inducing_points.shape[1]
        self.inducing_points = inducing_points
        self.training_points = training_points
        self.log_shift = log_shift
        self.diagonal_regularisation = diagonal_regularisation
        self.is_diagonal_regularisation_absolute_scale = (
            is_diagonal_regularisation_absolute_scale
        )
        self.reference_gram_inducing = self.calculate_reference_gram(x=inducing_points)
        self.reference_gram_inducing_cholesky_decomposition_and_lower = cho_factor(
            add_diagonal_regulariser(
                matrix=self.reference_gram_inducing,
                diagonal_regularisation=diagonal_regularisation,
                is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
            )
        )
        self.gram_inducing_train = self.calculate_reference_gram(
            x=inducing_points, y=training_points
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(self, parameters: Union[FrozenDict, Dict]) -> Parameters:
        """
        Generates a Pydantic model of the parameters for Stochastic Variational Gaussian Process Kernels.

        Args:
            parameters: A dictionary of the parameters for Stochastic Variational Gaussian Process Kernels.

        Returns: A Pydantic model of the parameters for Stochastic Variational Gaussian Process Kernels.

        """
        return StochasticVariationalGaussianProcessKernel.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey = None,
    ) -> StochasticVariationalGaussianProcessKernelParameters:
        """
        Initialise each parameter of the Stochastic Variational Gaussian Process Kernel with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters. Not required in this case becasue
                the parameters are initialised deterministically.

        Returns: A Pydantic model of the parameters for Stochastic Variational Gaussian Process Kernels.

        """
        # raise warning if key is not None

        return StochasticVariationalGaussianProcessKernel.Parameters(
            log_el_matrix=self.initialise_log_el_matrix_matrix(),
        )

    def initialise_log_el_matrix_matrix(self) -> jnp.ndarray:
        """
        Initialise the L matrix where:
            sigma_matrix = L @ L.T

        Returns: The L matrix.

        """
        reference_gaussian_measure_observation_precision = 1 / jnp.exp(
            self.reference_gaussian_measure_parameters.log_observation_noise
        )
        cholesky_decomposition_and_lower = cho_factor(
            jnp.linalg.cholesky(
                add_diagonal_regulariser(
                    matrix=(
                        self.reference_gram_inducing
                        + reference_gaussian_measure_observation_precision
                        * self.gram_inducing_train
                        @ self.gram_inducing_train.T
                    ),
                    diagonal_regularisation=self.diagonal_regularisation,
                    is_diagonal_regularisation_absolute_scale=self.is_diagonal_regularisation_absolute_scale,
                )
            )
        )
        el_matrix = cho_solve(
            c_and_lower=cholesky_decomposition_and_lower,
            b=jnp.eye(self.reference_gram_inducing.shape[0]),
        )
        el_matrix = jnp.clip(
            el_matrix,
            a_min=self.diagonal_regularisation,
            a_max=None,
        )
        return jnp.log(el_matrix)

    def calculate_sigma_matrix(self, log_el_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the sigma matrix where:
            sigma_matrix = L @ L.T

        Args:
            log_el_matrix: the log of the L matrix.

        Returns: The sigma matrix.

        """
        el_matrix = jnp.exp(log_el_matrix)

        # ensure lower triangle matrix
        el_matrix = jnp.tril(el_matrix)

        # add regularisation
        el_matrix = add_diagonal_regulariser(
            matrix=el_matrix,
            diagonal_regularisation=self.diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=self.is_diagonal_regularisation_absolute_scale,
        )

        # clip values to ensure greater than or equal to zero
        el_matrix = jnp.clip(
            el_matrix,
            a_min=self.diagonal_regularisation,
            a_max=None,
        )

        return el_matrix @ el_matrix.T

    def _calculate_gram(
        self,
        parameters: StochasticVariationalGaussianProcessKernelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        # pass because calculate_gram is overridden
        pass

    def calculate_gram(
        self,
        parameters: StochasticVariationalGaussianProcessKernelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Computing the Gram matrix using for the SVGP which depends on the reference kernel.

        If y is None, the Gram matrix is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the kernel gram matrix of shape (n, m)

        """
        Module.check_parameters(parameters, self.Parameters)
        x = jnp.atleast_2d(x)

        reference_gram_x_inducing = self.calculate_reference_gram(
            x=x,
            y=self.inducing_points,
        )

        # if y is None, compute for x and x
        if y is None:
            y = x
            reference_gram_y_inducing = reference_gram_x_inducing
        else:
            y = jnp.atleast_2d(y)
            self.check_inputs(x, y)

            reference_gram_y_inducing = self.calculate_reference_gram(
                x=y,
                y=self.inducing_points,
            )

        reference_gram_x_y = self.calculate_reference_gram(x=x, y=y)
        sigma_matrix = self.calculate_sigma_matrix(
            log_el_matrix=parameters.log_el_matrix
        )
        return (
            reference_gram_x_y
            - reference_gram_x_inducing
            @ cho_solve(
                c_and_lower=self.reference_gram_inducing_cholesky_decomposition_and_lower,
                b=reference_gram_y_inducing.T,
            )
            + reference_gram_x_inducing @ sigma_matrix @ reference_gram_y_inducing.T
        )
