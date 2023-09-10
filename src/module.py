from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Type, Union

import jax.numpy as jnp
import orbax
import pydantic
from flax.core.frozen_dict import FrozenDict
from flax.training import orbax_utils
from pydantic import BaseModel

from src.utils.custom_types import JSON_ENCODERS

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
PYDANTIC_VALIDATION_CONFIG = dict(arbitrary_types_allowed=True)


class ModuleParameters(BaseModel, ABC):
    """
    A base class for parameters. All model parameter classes will inherit this ABC.
    """

    class Config:
        json_encoders = JSON_ENCODERS

    def save(self, path: str) -> None:
        """
        Save the parameters as a checkpoint.

        Args:
            path: save path

        """
        ckpt = self.dict()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            path,
            ckpt,
            save_args=save_args,
            force=True,
        )

    def load(self, path: str) -> ModuleParameters:
        """
        Load existing parameters.

        Args:
            path: parameter checkpoint path

        Returns: Module parameters with loaded parameters

        """
        ckpt = orbax_checkpointer.restore(path)
        return self.construct(**ckpt)


class Module(ABC):
    """
    A base class for all models. All model classes will inheret this ABC.
    """

    # every model must have a corresponding Pydantic class of module parameters
    Parameters: ModuleParameters = ModuleParameters

    def __init__(
        self, preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None
    ):
        """
        Construct for the Module class.
        Args:
            preprocess_function: a function to preprocess the inputs of the kernel function,
                                 defaults to None (identity)
        """
        if preprocess_function is None:
            self.preprocess_function = lambda x: x
        else:
            self.preprocess_function = preprocess_function

    @staticmethod
    def check_parameters(
        parameters: ModuleParameters, parameter_type: Type[ModuleParameters]
    ) -> None:
        """
        Checks that the parameters are valid.

        Args:
            parameters: A Pydantic model of the parameters for the Module.
            parameter_type: The type of the parameters for the Module.

        """
        assert isinstance(
            parameters, parameter_type
        ), f"Parameters is type: {type(parameters)=}, needs to be {parameter_type=}"

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    @abstractmethod
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> ModuleParameters:
        """
        Generates a Pydantic model of the parameters for the Module.

        Args:
            parameters: A dictionary of the parameters for the Module.

        Returns: A Pydantic model of the parameters for the Module.

        """
        raise NotImplementedError
