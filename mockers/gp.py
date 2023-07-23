from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import pydantic
from flax.core import FrozenDict

from mockers.kernel import MockKernel, MockKernelParameters
from mockers.mean import MockMean, MockMeanParameters
from src.distributions import Distribution
from src.gps.base.base import GPBase, GPBaseParameters
from src.utils.custom_types import JaxFloatType

PRNGKey = Any  # pylint: disable=invalid-name


class MockGPParameters(GPBaseParameters):
    log_observation_noise: JaxFloatType = jnp.log(1.0)
    mean: MockMeanParameters = MockMeanParameters()
    kernel: MockKernelParameters = MockKernelParameters()


class MockGP(GPBase):
    Parameters = MockGPParameters

    def __init__(self):
        GPBase.__init__(
            self,
            mean=MockMean(),
            kernel=MockKernel(),
        )

    def _calculate_prediction_gaussian(
        self,
        parameters: MockGPParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            self.mean.predict(parameters=parameters.mean, x=x),
            self.kernel.calculate_gram(
                parameters=parameters.kernel,
                x1=x,
                x2=x,
                full_covariance=full_covariance,
            ),
        )

    def _calculate_prediction_gaussian_covariance(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        return self.kernel.calculate_gram(
            parameters=parameters.kernel,
            x1=x,
            x2=x,
            full_covariance=full_covariance,
        )

    def _construct_distribution(
        self,
        probabilities: Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        full_covariance: bool = False,
    ) -> Distribution:
        pass

    def _predict_probability(
        self,
        parameters: MockGPParameters,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._calculate_prediction_gaussian(
            parameters=parameters,
            x=x,
            full_covariance=False,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> MockGPParameters:
        """
        Generates a Pydantic model of the parameters for Reference Gaussian Measures.

        Args:
            parameters: A dictionary of the parameters for Reference Gaussian Measures.

        Returns: A Pydantic model of the parameters for Reference Gaussian Measures.

        """
        return MockGP.Parameters(
            log_observation_noise=parameters["log_observation_noise"],
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> MockGPParameters:
        """
        Initialise each parameter of the Reference Gaussian measure with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Reference Gaussian Measures.

        """
        pass
