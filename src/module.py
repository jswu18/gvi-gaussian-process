from abc import ABC, abstractmethod
from typing import Any

from flax.core.frozen_dict import FrozenDict

from src.parameters.module import ModuleParameters

PRNGKey = Any  # pylint: disable=invalid-name


class Module(ABC):
    Parameters: ModuleParameters = ModuleParameters

    @abstractmethod
    def generate_parameters(self, parameters: FrozenDict) -> Parameters:
        """
        Generates a Pydantic model of the parameters for the Module.

        Args:
            parameters: A dictionary of the parameters for the Module.

        Returns: A Pydantic model of the parameters for the Module.

        """
        raise NotImplementedError

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
