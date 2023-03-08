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
        Generator for a Pydantic model of the parameters for the module.
        Args:
            parameters: A dictionary of the parameters of the module.

        Returns: A Pydantic model of the parameters for the module.

        """
        raise NotImplementedError

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ModuleParameters:
        """
        Initialise the parameters of the module using a random key.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        raise NotImplementedError
