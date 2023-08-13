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


class MLPGPKernelParameters(NNGPKernelBaseParameters):
    w_std: JaxArrayType[Literal["float64"]]
    b_std: JaxArrayType[Literal["float64"]]


class MLPGPKernel(NNGPKernelBase):
    Parameters = MLPGPKernelParameters

    def __init__(self, features: Sequence[int]):
        self.features = features
        super().__init__()

    def initialise_parameters(self):
        return self.Parameters(
            w_std=jnp.ones((len(self.features),)),
            b_std=jnp.ones(
                (len(self.features)),
            ),
        )

    def __call__(
        self,
        parameters: Union[Dict, FrozenDict, MLPGPKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        if not isinstance(parameters, self.Parameters):
            parameters = MLPGPKernelParameters(
                w_std=parameters["w_std"],
                b_std=parameters["b_std"],
            )
        nn_architecture = []
        for i, feat in enumerate(self.features[:-1]):
            nn_architecture.append(
                stax.Dense(
                    out_dim=feat, W_std=parameters.w_std[i], b_std=parameters.b_std[i]
                )
            )
            nn_architecture.append(stax.Erf())
        nn_architecture.append(
            stax.Dense(
                out_dim=self.features[-1],
                W_std=parameters.w_std[-1],
                b_std=parameters.b_std[-1],
                parameterization="standard",
            )
        )
        _, _, kernel_fn = stax.serial(*nn_architecture)
        return kernel_fn(x1, x2, "nngp")


class CNNGPKernelParameters(NNGPKernelBaseParameters):
    w_std: JaxArrayType[Literal["float64"]]
    b_std: JaxArrayType[Literal["float64"]]


class CNNGPKernel(NNGPKernelBase):
    """
    NNGP kernel for example CNN architecture from
    https://flax.readthedocs.io/en/latest/getting_started.html
    """

    Parameters = CNNGPKernelParameters

    def __init__(self, number_of_outputs: int):
        self.number_of_outputs = number_of_outputs
        super().__init__()

    def initialise_parameters(self):
        return self.Parameters(
            w_std=jnp.ones((4,)),
            b_std=jnp.ones((4,)),
        )

    def __call__(
        self,
        parameters: Union[Dict, FrozenDict, CNNGPKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        if not isinstance(parameters, self.Parameters):
            parameters = CNNGPKernelParameters(
                w_std=parameters["w_std"],
                b_std=parameters["b_std"],
            )
        nn_architecture = [
            stax.Conv(
                out_chan=32,
                filter_shape=(3, 3),
                W_std=parameters.w_std[0],
                b_std=parameters.b_std[0],
                parameterization="standard",
            ),
            stax.Relu(),
            stax.AvgPool(
                window_shape=(2, 2),
                strides=(2, 2),
            ),
            stax.Conv(
                out_chan=64,
                filter_shape=(3, 3),
                W_std=parameters.w_std[1],
                b_std=parameters.b_std[1],
                parameterization="standard",
            ),
            stax.Relu(),
            stax.AvgPool(
                window_shape=(2, 2),
                strides=(2, 2),
            ),
            stax.Flatten(),
            stax.Dense(
                out_dim=256,
                W_std=parameters.w_std[2],
                b_std=parameters.b_std[2],
                parameterization="standard",
            ),
            stax.Relu(),
            stax.Dense(
                out_dim=self.number_of_outputs,
                W_std=parameters.w_std[3],
                b_std=parameters.b_std[3],
                parameterization="standard",
            ),
        ]
        _, _, kernel_fn = stax.serial(*nn_architecture)
        return kernel_fn(x1, x2, "nngp")
