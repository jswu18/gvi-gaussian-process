from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core import FrozenDict

from src.module import PYDANTIC_VALIDATION_CONFIG, Module, ModuleParameters
from src.utils.custom_types import JaxFloatType


class ConformalRegressionBaseParameters(ModuleParameters, ABC):
    """
    There are no learnable parameters for conformal regression.
    """


class ConformalRegressionBase(Module, ABC):
    """
    A base class for all conformal regression models.
    """

    Parameters = ConformalRegressionBaseParameters

    def __init__(
        self,
        x_calibration: jnp.ndarray,
        y_calibration: jnp.ndarray,
    ):
        """
        Construct for the ConformalRegressionBase class.
        Args:
            x_calibration: input data used for calibration of shape (n, d)
            y_calibration: response data used for calibration of shape (n, 1)
        """
        self.x_calibration = x_calibration
        self.y_calibration = y_calibration
        self.number_of_calibration_points = x_calibration.shape[0]
        self._jit_compiled_predict_coverage = jax.jit(
            lambda x, coverage: self._predict_coverage(x=x, coverage=coverage)
        )
        super().__init__(preprocess_function=None)

    @abstractmethod
    def _predict_uncalibrated_coverage(
        self,
        x: jnp.ndarray,
        coverage: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns uncalibrated coverage predictions for a given input and coverage percentage
        Args:
            x: input data of shape (n, d)
            coverage: the coverage percentage

        Returns: Tuple of lower and upper bounds of shape (1, n)

        """
        raise NotImplementedError

    @abstractmethod
    def _predict_median(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Returns median predictions for a given input.
        Args:
            x: input data of shape (n, d)

        Returns: median predictions of shape (1, n)

        """
        raise NotImplementedError

    def _calculate_calibration(self, coverage: float) -> JaxFloatType:
        """
        Calculates the calibration factor for a given coverage percentage.
        Args:
            coverage: the coverage percentage

        Returns: the calibration factor calculated with conformal prediction calibration

        """
        uncalibrated_lower, uncalibrated_upper = self._predict_uncalibrated_coverage(
            x=self.x_calibration,
            coverage=coverage,
        )
        scores = jnp.max(
            jnp.concatenate(
                [
                    uncalibrated_lower - self.y_calibration,
                    self.y_calibration - uncalibrated_upper,
                ],
                axis=0,
            ),
            axis=0,
        )
        calibration = jnp.quantile(
            scores,
            jnp.clip(
                (self.number_of_calibration_points + 1)
                * coverage
                / self.number_of_calibration_points,
                0.0,
                1.0,
            ),
        )
        return jnp.float64(calibration)

    def _predict_coverage(
        self, x: jnp.ndarray, coverage: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predicts the calibrated coverage of a given input x at a given coverage percentage.
        Args:
            x: input data of shape (n, d)
            coverage: the coverage percentage

        Returns: Tuple of lower and upper bounds of shape (1, n)

        """
        calibration = self._calculate_calibration(coverage)
        uncalibrated_lower, uncalibrated_upper = self._predict_uncalibrated_coverage(
            x=x, coverage=coverage
        )
        calibrated_lower, calibrated_upper = (
            uncalibrated_lower - calibration,
            uncalibrated_upper + calibration,
        )
        median = self._predict_median(x)
        # nothing should cross the mean
        return (
            jnp.min(jnp.concatenate([calibrated_lower, median], axis=0), axis=0),
            jnp.max(jnp.concatenate([calibrated_upper, median], axis=0), axis=0),
        )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def predict_coverage(
        self, x: jnp.ndarray, coverage: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calls the jit-compiled method to calculate the calibrated
        coverage of a given input x at a given coverage percentage.
        Args:
            x: input data of shape (n, d)
            coverage: the coverage percentage

        Returns: Tuple of lower and upper bounds of shape (1, n)

        """
        return self._jit_compiled_predict_coverage(x, coverage)

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def calculate_average_interval_width(
        self, x: jnp.ndarray, coverage: float
    ) -> float:
        """
        Calculates the average interval width of a given input x at a given coverage percentage.
        Args:
            x: input data of shape (n, d)
            coverage: the coverage percentage

        Returns: the average interval width

        """
        lower, upper = self.predict_coverage(x=x, coverage=coverage)
        return float(jnp.mean(upper - lower))

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict] = None
    ) -> ModuleParameters:
        return ConformalRegressionBaseParameters()
