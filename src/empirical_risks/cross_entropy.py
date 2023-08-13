import jax.numpy as jnp
import jax_metrics as jm

from src.distributions import Multinomial
from src.empirical_risks.base import EmpiricalRiskBase
from src.gps.base.base import GPBaseParameters
from src.gps.base.classification_base import GPClassificationBase
from src.utils.custom_types import JaxFloatType


class CrossEntropy(EmpiricalRiskBase):
    def __init__(self, gp: GPClassificationBase):
        self.cross_entropy = jm.losses.Crossentropy()
        super().__init__(gp)

    def _calculate_empirical_risk(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> JaxFloatType:
        multinomial = Multinomial(
            **self.gp.predict_probability(parameters=parameters, x=x).dict()
        )
        return jnp.float64(
            self.cross_entropy(
                target=y,
                preds=multinomial.probabilities,
            )
        )
