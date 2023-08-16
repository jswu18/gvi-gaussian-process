import optax

from experiments.shared.schemas import OptimiserSchema


def optimiser_resolver(
    optimiser_schema: OptimiserSchema, learning_rate: float
) -> optax.GradientTransformation:
    if optimiser_schema == OptimiserSchema.adam:
        return optax.adam(learning_rate=learning_rate)
    elif optimiser_schema == OptimiserSchema.adabelief:
        return optax.adabelief(learning_rate=learning_rate)
    elif optimiser_schema == OptimiserSchema.rmsprop:
        return optax.rmsprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimiser: {optimiser_schema=}")
