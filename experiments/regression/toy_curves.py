from abc import ABC, abstractmethod

from jax import numpy as jnp
from jax import random

from src.utils.custom_types import PRNGKey


class Curve(ABC):
    seed: int

    @abstractmethod
    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        raise NotImplementedError


class Curve0(Curve):
    __name__ = "$y=2 \sin(0.35 \pi (x-3)^2) + x^2$"
    seed: int = 0

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(((x - 3) ** 2) * 0.35 * jnp.pi)
            + x**2
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve1(Curve):
    __name__ = "$y=2\sin(\pi x)$"
    seed: int = 1

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(x * jnp.pi) + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve2(Curve):
    __name__ = "$y=1.2 \cos(2 \pi x)$ + x^2"
    seed: int = 2

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            1.2 * jnp.cos(x * (2 * jnp.pi))
            + x**2
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve3(Curve):
    __name__ = "$y=\sin(1.5\pi x) + 0.3 \cos(4.5 \pi x) + 0.5 \sin(3.5 \pi x)$"
    seed: int = 3

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            jnp.sin(x * (1.5 * jnp.pi))
            + 0.3 * jnp.cos(x * (4.5 * jnp.pi))
            + 0.5 * jnp.sin(x * (3.5 * jnp.pi))
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve4(Curve):
    __name__ = "$y=2 \sin(\pi x) + x$"
    seed: int = 4

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(jnp.pi * x) + x + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve5(Curve):
    __name__ = "$y=\sin(\pi x) + 0.3x^3$"
    seed: int = 5

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            jnp.sin(jnp.pi * x)
            + 0.3 * (x**3)
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve6(Curve):
    __name__ = "$y=2\sin(\pi x) + \sin(3 \pi x) -x$"
    seed: int = 6

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(x * jnp.pi)
            + jnp.sin(x * (3 * jnp.pi))
            - x
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve7(Curve):
    __name__ = "$y=2\cos(\pi x) + \sin(3 \pi x) -x^2$"
    seed: int = 7

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.cos(x * jnp.pi)
            + jnp.sin(x * (3 * jnp.pi))
            - x**2
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve8(Curve):
    __name__ = "$y=\sin(0.5 \pi (x-2)^2)$"
    seed: int = 8

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            jnp.sin(((x - 2) ** 2) * 0.5 * jnp.pi)
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve9(Curve):
    __name__ = "$y=3\sqrt{4-x^2} + \sin(\pi x)$"
    seed: int = 9

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            3 * jnp.sqrt(4 - x**2)
            + jnp.sin(jnp.pi * x)
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


CURVE_FUNCTIONS = [
    Curve0(),
    Curve1(),
    Curve2(),
    Curve3(),
    Curve4(),
    Curve5(),
    Curve6(),
    Curve7(),
    Curve8(),
    Curve9(),
]
