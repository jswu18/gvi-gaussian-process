import optax

from experiments.shared.schemes import OptimiserScheme


def optimiser_resolver(
    optimiser_scheme: OptimiserScheme, learning_rate: float
) -> optax.GradientTransformation:
    if optimiser_scheme == OptimiserScheme.adam:
        return optax.adam(learning_rate=learning_rate)
    elif optimiser_scheme == OptimiserScheme.adabelief:
        return optax.adabelief(learning_rate=learning_rate)
    elif optimiser_scheme == OptimiserScheme.rmsprop:
        return optax.rmsprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimiser: {optimiser_scheme=}")
