import warnings
from typing import Any

import jax.numpy as jnp
from jax import random

PRNGKey = Any  # pylint: disable=invalid-name


def sample_discrete_unnormalised_distribution(
    key: PRNGKey, unnormalised_probabilities: jnp.ndarray
) -> int:
    """
    Sample from a discrete distribution with unnormalised probabilities. If all of the probabilities are numerically 0,
    sample uniformly.
    Adapted from https://github.com/markvdw/RobustGP/blob/master/robustgp/init_methods/misc.py

    Args:
        key: PRNGKey, random key
        unnormalised_probabilities: jnp.ndarray, unnormalised probabilities

    Returns: int, index of sampled point

    """
    unnormalised_probabilities = jnp.clip(unnormalised_probabilities, 0, None)
    number_of_points = unnormalised_probabilities.shape[0]
    normalization = jnp.sum(unnormalised_probabilities)
    if (
        normalization == 0
    ):  # if all of the probabilities are numerically 0, sample uniformly
        warnings.warn("Trying to sample discrete distribution with all 0 weights")
        return int(random.choice(key=key, a=number_of_points, shape=(1,))[0])
    normalised_probabilities = unnormalised_probabilities / normalization
    return int(
        random.choice(
            key=key, a=number_of_points, shape=(1,), p=normalised_probabilities
        )[0]
    )
