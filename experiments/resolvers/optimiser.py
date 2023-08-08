import optax

from experiments.schemes import Optimiser


def optimiser(
    optimiser_scheme: Optimiser, learning_rate: float
) -> optax.GradientTransformation:
    if optimiser_scheme == Optimiser.adam:
        return optax.adam(learning_rate=learning_rate)
    elif optimiser_scheme == Optimiser.adabelief:
        return optax.adabelief(learning_rate=learning_rate)
    elif optimiser_scheme == Optimiser.rmsprop:
        return optax.rmsprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimiser: {optimiser_scheme=}")
