from dataclasses import asdict, dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from src.gaussian_measure import GaussianMeasure, StochasticGaussianProcess
from src.parameters import Parameters


@dataclass
class GaussianWassersteinDistanceFrozenParameters(Parameters):
    gaussian_measure_p_mean: Dict[str, Any]
    gaussian_measure_p_kernel: Dict[str, Any]
    gaussian_measure_p_log_sigma: float
    gaussian_measure_q_kernel: Dict[str, Any]
    gaussian_measure_q_mean_shift: float = 0


@dataclass
class GaussianWassersteinDistanceVariableParameters(Parameters):
    gaussian_measure_q_mean: Dict[str, Any]
    gaussian_measure_q_beta: Dict[str, Any]


class GaussianWassersteinDistance:

    FrozenParameters = GaussianWassersteinDistanceFrozenParameters
    VariableParameters = GaussianWassersteinDistanceVariableParameters

    def __init__(
        self,
        gaussian_measure_p: GaussianMeasure,
        gaussian_measure_q: StochasticGaussianProcess,
        eigenvalue_regularisation: float = 100,
    ):
        self.gaussian_measure_p = gaussian_measure_p
        self.gaussian_measure_q = gaussian_measure_q
        self.eigenvalue_regularisation = eigenvalue_regularisation

    @staticmethod
    def calculate_mean_squared_distance(a: jnp.ndarray, b: jnp.ndarray) -> jnp.float64:
        return ((a.reshape(-1) - b.reshape(-1)) ** 2).mean()

    @staticmethod
    def calculate_mean_diagonal(matrix: jnp.ndarray) -> jnp.float64:
        return jnp.diagonal(matrix).mean()

    @staticmethod
    def calculate_eigenvalues(
        matrix: jnp.ndarray, regularisation: float
    ) -> jnp.ndarray:
        eigen_values, _ = jnp.linalg.eigh(
            matrix + regularisation * jnp.eye(matrix.shape[0])
        )
        return eigen_values - regularisation

    def calculate(
        self,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
        x_sample: jnp.ndarray,
        data_set_size: int,
        frozen_parameters,
        variable_parameters,
        predict_function=None,
    ) -> jnp.float64:
        frozen_parameters = self.FrozenParameters(**frozen_parameters)
        variable_parameters = self.VariableParameters(**variable_parameters)
        if predict_function is None:
            gaussian_measure_parameters_p = self.gaussian_measure_p.Parameters(
                mean=frozen_parameters.gaussian_measure_p_mean,
                log_sigma=frozen_parameters.gaussian_measure_p_log_sigma,
                kernel=frozen_parameters.gaussian_measure_p_kernel,
            )
            gaussian_measure_parameters_q = self.gaussian_measure_q.Parameters(
                mean=variable_parameters.gaussian_measure_q_mean,
                log_sigma=frozen_parameters.gaussian_measure_p_log_sigma,
                kernel=frozen_parameters.gaussian_measure_q_kernel,
                beta=variable_parameters.gaussian_measure_q_beta,
            )
        else:
            gaussian_measure_parameters_p = self.gaussian_measure_p.Parameters(
                mean=frozen_parameters.gaussian_measure_p_mean,
                log_sigma=frozen_parameters.gaussian_measure_p_log_sigma,
                predict_function=predict_function,
                kernel=frozen_parameters.gaussian_measure_p_kernel,
            )
            gaussian_measure_parameters_q = self.gaussian_measure_q.Parameters(
                mean=variable_parameters.gaussian_measure_q_mean,
                log_sigma=frozen_parameters.gaussian_measure_p_log_sigma,
                kernel=frozen_parameters.gaussian_measure_q_kernel,
                predict_function=predict_function,
                beta=variable_parameters.gaussian_measure_q_beta,
                mean_shift=frozen_parameters.gaussian_measure_q_mean_shift,
            )
        if isinstance(gaussian_measure_parameters_p, dict):
            gaussian_measure_parameters_p = self.gaussian_measure_p.Parameters(
                **gaussian_measure_parameters_p
            )
        if isinstance(gaussian_measure_parameters_q, dict):
            gaussian_measure_parameters_q = self.gaussian_measure_q.Parameters(
                **gaussian_measure_parameters_q
            )
        batch_size = x_batch.shape[0]
        sample_size = x_sample.shape[0]
        batch_mean_p, batch_covariance_p = self.gaussian_measure_p.mean_and_covariance(
            x=x_batch,
            **asdict(gaussian_measure_parameters_p),
        )
        batch_mean_q, batch_covariance_q = self.gaussian_measure_q.mean_and_covariance(
            x=x_batch,
            **asdict(gaussian_measure_parameters_q),
        )
        k_batch_sample_p = self.gaussian_measure_p.kernel(
            x_batch, x_sample, **gaussian_measure_parameters_p.kernel
        )
        k_batch_sample_q = self.gaussian_measure_q.kernel(
            x_batch, x_sample, **gaussian_measure_parameters_q.kernel
        )

        mean_squared_error = self.calculate_mean_squared_distance(y_batch, batch_mean_q)
        mean_squared_distance = self.calculate_mean_squared_distance(
            batch_mean_p, batch_mean_q
        )
        mean_p_gram_diagonal = self.calculate_mean_diagonal(batch_covariance_p)
        mean_q_gram_diagonal = self.calculate_mean_diagonal(batch_covariance_q)
        covariance_eigenvalues = (
            self.calculate_eigenvalues(  # TODO need better name for this
                k_batch_sample_p.T @ k_batch_sample_q,
                regularisation=self.eigenvalue_regularisation,
            )
        )

        return (
            (0.5 * data_set_size * gaussian_measure_parameters_p.precision)
            * (mean_squared_error + mean_q_gram_diagonal)
            + mean_squared_distance
            + mean_p_gram_diagonal
            + mean_q_gram_diagonal
            - (2 / np.sqrt(batch_size * sample_size))
            * jnp.mean(jnp.sqrt(covariance_eigenvalues))
        ).real

    def _compute_gradient(self, **parameter_args) -> Dict[str, Any]:
        """Calculate the gradient of the posterior negative log likelihood with respect to the parameters.

        Args:
            **parameter_args: parameter arguments for the Gaussian Process

        Returns:
            A dictionary of the gradients for each parameter argument.
        """
        gradients = jax.grad(lambda params: self.calculate(**params))(parameter_args)
        return gradients
