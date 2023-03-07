from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Dict

from flax.core.frozen_dict import FrozenDict

from src import decorators

PRNGKey = Any  # pylint: disable=invalid-name


class Module(ABC):
    parameter_keys: Dict[str, type] = NotImplementedError

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

    # @decorators.common.check_parameters(parameter_keys)
    def initialise_parameters(self, parameters: Dict[str, type]) -> FrozenDict:
        """
        Initialise the parameters of the module using the provided arguments.
        Args:
            # **kwargs: The parameters of the module.

        Returns: A dictionary of the parameters of the module.

        """
        print(self.parameter_keys)
        return FrozenDict(
            {
                parameter_key: parameters[parameter_key]
                for parameter_key in self.parameter_keys
            }
        )
