from abc import ABC, abstractmethod
from typing import Any

from flax.core.frozen_dict import FrozenDict

PRNGKey = Any  # pylint: disable=invalid-name


class Module(ABC):
    @abstractmethod
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        """
        Initialise the parameters of the module using a random key.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        raise NotImplementedError

    @abstractmethod
    def initialise_parameters(self, **kwargs) -> FrozenDict:
        """
        Initialise the parameters of the module using the provided arguments.
        Args:
            **kwargs: The parameters of the module.

        Returns: A dictionary of the parameters of the module.

        """
        raise NotImplementedError
