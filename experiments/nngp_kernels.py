from abc import ABC, abstractmethod
from typing import Dict, Literal, Sequence, Union

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from neural_tangents import stax

from src.module import ModuleParameters
from src.utils.custom_types import JaxArrayType


class NNGPKernelBaseParameters(ModuleParameters, ABC):
    pass


class NNGPKernelBase(ABC):
    Parameters = NNGPKernelBaseParameters

    @abstractmethod
    def initialise_parameters(self) -> NNGPKernelBaseParameters:
        pass

    @abstractmethod
    def __call__(
        self, parameters: NNGPKernelBaseParameters, x1: jnp.ndarray, x2: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError


class MultiLayerPerceptronKernelParameters(NNGPKernelBaseParameters):
    w_std: JaxArrayType[Literal["float64"]]
    b_std: JaxArrayType[Literal["float64"]]


class MultiLayerPerceptronKernel(NNGPKernelBase):
    Parameters = MultiLayerPerceptronKernelParameters

    def __init__(self, features: Sequence[int]):
        self.features = features
        super().__init__()

    def initialise_parameters(self):
        return self.Parameters(
            w_std=jnp.ones(len(self.features)).astype(jnp.float64),
            b_std=jnp.ones(len(self.features)).astype(jnp.float64),
        )

    def __call__(
        self,
        parameters: Union[Dict, FrozenDict, MultiLayerPerceptronKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        if not isinstance(parameters, self.Parameters):
            parameters = MultiLayerPerceptronKernelParameters(**parameters)
        nn_architecture = []
        for i, feat in enumerate(self.features):
            nn_architecture.append(
                stax.Dense(feat, W_std=parameters.w_std[i], b_std=parameters.b_std[i])
            )
            nn_architecture.append(stax.Erf())
        nn_architecture.pop(-1)
        if not isinstance(parameters, self.Parameters):
            parameters = MultiLayerPerceptronKernelParameters(
                w_std=parameters["w_std"],
                b_std=parameters["b_std"],
            )
        _, _, kernel_fn = stax.serial(*nn_architecture)
        return kernel_fn(x1, x2, "nngp")
