from abc import ABC
from typing import Dict, Tuple, Union

import jax.numpy as jnp
import pydantic
from flax.core import FrozenDict

from src.gps.base.base import GPBase, GPBaseParameters
from src.kernels.base import KernelBase
from src.means.base import MeanBase
from src.utils.custom_types import JaxFloatType


class ApproximateGPBaseParameters(GPBaseParameters):
    log_observation_noise: JaxFloatType = jnp.log(0)


class ApproximateGPBase(GPBase, ABC):
    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
    ):
        GPBase.__init__(self, mean=mean, kernel=kernel)

    def _calculate_prediction_gaussian(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.calculate_prior(
            parameters=parameters, x=x, full_covariance=full_covariance
        )

    def _calculate_prediction_gaussian_covariance(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        return self.calculate_prior_covariance(
            parameters=parameters, x=x, full_covariance=full_covariance
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ApproximateGPBaseParameters:
        pass
