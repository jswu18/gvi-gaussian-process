import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import pydantic

from src.distributions import Multinomial
from src.gps.base.base import GPBaseParameters
from src.gps.gp_classification import GPClassificationBase
from src.regularisations.base import RegularisationBase


class MultinomialWassersteinRegularisation(RegularisationBase):
    def __init__(
        self,
        gp: GPClassificationBase,
        regulariser: GPClassificationBase,
        regulariser_parameters: GPBaseParameters,
        power: int = 2,
    ):
        self.power = power
        super().__init__(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )

    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> jnp.float64:
        multinomial_p = Multinomial(
            **self.regulariser.predict_probability(
                parameters=self.regulariser_parameters,
                x=x,
            ).dict()
        )
        multinomial_q = Multinomial(
            **self.gp.predict_probability(
                parameters=parameters,
                x=x,
            ).dict()
        )
        return jnp.float64(
            jnp.mean(
                jnp.power(
                    jnp.sum(
                        jnp.power(
                            jnp.abs(
                                multinomial_p.probabilities
                                - multinomial_q.probabilities
                            ),
                            self.power,
                        ),
                        axis=1,
                    ),
                    1 / self.power,
                )
            )
        )
