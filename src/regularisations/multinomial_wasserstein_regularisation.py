import jax.numpy as jnp

from src.distributions import Multinomial
from src.gps.base.base import GPBaseParameters
from src.gps.gp_classification import GPClassificationBase
from src.regularisations.base import RegularisationBase
from src.regularisations.schemas import RegularisationMode


class MultinomialWassersteinRegularisation(RegularisationBase):
    """
    A regulariser which is the Wasserstein distance between the discrete probabilities
    of two Multinomial distributions.
    """

    def __init__(
        self,
        gp: GPClassificationBase,
        regulariser: GPClassificationBase,
        regulariser_parameters: GPBaseParameters,
        mode: RegularisationMode = RegularisationMode.prior,
        power: int = 2,
    ):
        self.power = power
        super().__init__(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            mode=mode,
        )

    def _calculate_regulariser_multinomial(
        self,
        x: jnp.ndarray,
    ) -> Multinomial:
        if self._mode == RegularisationMode.prior:
            raise NotImplementedError
        elif self._mode == RegularisationMode.posterior:
            return Multinomial(
                **self.regulariser.predict_probability(
                    parameters=self.regulariser_parameters,
                    x=x,
                ).dict()
            )

    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> jnp.float64:
        multinomial_p = self._calculate_regulariser_multinomial(
            x=x,
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
