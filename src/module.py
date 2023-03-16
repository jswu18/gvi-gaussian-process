from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Union

import pydantic
from flax.core.frozen_dict import FrozenDict

from src.parameters.module import ModuleParameters

PRNGKey = Any  # pylint: disable=invalid-name


class Module(ABC):
    Parameters: ModuleParameters = ModuleParameters

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

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
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

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ModuleParameters:
        """
        Initialise each parameter of the Module with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for the Module.

        """
        raise NotImplementedError
