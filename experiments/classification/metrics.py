import jax.numpy as jnp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from experiments.classification.schemas import ClassificationMetricSchema
from experiments.shared.data import Data
from src.distributions import Multinomial
from src.gps.base.base import GPBaseParameters
from src.gps.gp_classification import GPClassificationBase


def calculate_metric(
    gp: GPClassificationBase,
    gp_parameters: GPBaseParameters,
    data: Data,
    metric_schema: ClassificationMetricSchema,
) -> float:
    y_pred = jnp.argmax(
        Multinomial(
            **gp.predict_probability(gp_parameters, data.x).dict()
        ).probabilities,
        axis=1,
    )
    y_true = jnp.argmax(data.y, axis=1)
    if metric_schema == ClassificationMetricSchema.accuracy:
        return accuracy_score(y_true, y_pred)
    elif metric_schema == ClassificationMetricSchema.f1:
        return f1_score(y_true, y_pred)
    elif metric_schema == ClassificationMetricSchema.precision:
        return precision_score(y_true, y_pred)
    elif metric_schema == ClassificationMetricSchema.recall:
        return recall_score(y_true, y_pred)
