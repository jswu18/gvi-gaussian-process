from abc import ABC, abstractmethod
from typing import Any

from jax import numpy as jnp
from jax import random

PRNGKey = Any  # pylint: disable=invalid-name


class Curve(ABC):
    @abstractmethod
    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        raise NotImplementedError


class Curve1(Curve):
    __name__ = "$y=2\sin(\pi x)$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(x * jnp.pi) + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve2(Curve):
    __name__ = "$y=1.2 \cos(\pi x)$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            1.2 * jnp.cos(x * (2 * jnp.pi))
            + jnp.square(x)
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve3(Curve):
    __name__ = "$y=\sin(1.5\pi x)$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            jnp.sin(x * (1.5 * jnp.pi))
            + 0.3 * jnp.cos(x * (4.5 * jnp.pi))
            + 0.5 * jnp.sin(x * (3.5 * jnp.pi))
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve4(Curve):
    __name__ = "$y=\sin(\pi x) + 0.1x$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(jnp.pi * x) + x + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve5(Curve):
    __name__ = "$y=\sin(\pi x) + 0.1x^2$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            jnp.sin(jnp.pi * x)
            + 0.3 * (x**3)
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


CURVE_FUNCTIONS = [
    Curve1(),
    Curve2(),
    Curve3(),
    Curve4(),
    Curve5(),
]
