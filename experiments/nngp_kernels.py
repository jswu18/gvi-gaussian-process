from abc import ABC, abstractmethod
from typing import Literal, Sequence

from jax.numpy import jnp
from neural_tangents import stax

from src.module import ModuleParameters
from src.utils.custom_types import JaxArrayType


class NNGPKernelParameters(ModuleParameters, ABC):
    pass


class NNGPKernelBase(ABC):
    Parameters = NNGPKernelParameters

    @abstractmethod
    def initialise_parameters(self) -> NNGPKernelParameters:
        pass

    @abstractmethod
    def __call__(
        self, parameters: NNGPKernelParameters, x1: jnp.ndarray, x2: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError


class MultiLayerPerceptronKernelParameters(NNGPKernelParameters):
    w_std: JaxArrayType[Literal["float64"]]
    b_std: JaxArrayType[Literal["float64"]]


class MultiLayerPerceptronKernel(NNGPKernelParameters):
    Parameters = MultiLayerPerceptronKernelParameters

    def __init__(self, features: Sequence[int]):
        self.features = features
        super().__init__()

    def initialise_parameters(self):
        return self.Parameters(
            w_std=jnp.ones(len(self.features)),
            b_std=jnp.ones(len(self.features)),
        )

    def __call__(
        self,
        parameters: MultiLayerPerceptronKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        _, _, kernel_fn = stax.serial(
            *[
                stax.Dense(feat, W_std=parameters.w_std[i], b_std=parameters.b_std[i])
                for i, feat in enumerate(self.features)
            ]
        )
        return kernel_fn(x1, x2, "nngp")
