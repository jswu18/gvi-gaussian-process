from experiments.shared.schemas import InducingPointsSelectorSchema
from src.inducing_points_selection import (
    ConditionalVarianceInducingPointsSelector,
    RandomInducingPointsSelector,
)


def inducing_points_selector_resolver(
    inducing_points_schema: InducingPointsSelectorSchema,
):
    if inducing_points_schema == InducingPointsSelectorSchema.random:
        return RandomInducingPointsSelector()
    if inducing_points_schema == InducingPointsSelectorSchema.conditional_variance:
        return ConditionalVarianceInducingPointsSelector()
    raise ValueError(f"Unknown inducing points scheme: {inducing_points_schema}")
