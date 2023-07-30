import flax
from jax import numpy as jnp


class MockNeuralNetwork(flax.linen.Module):
    def init(self, *args, **kwargs):
        return None

    def apply(self, variables, *args, **kwargs):
        if "x" in kwargs:
            return jnp.ones((kwargs["x"].shape[0], 1))
        else:
            return jnp.ones((args[0].shape[0], 1))
