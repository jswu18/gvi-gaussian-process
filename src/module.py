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
        raise NotImplementedError

    @abstractmethod
    def initialise_parameters(self, **kwargs) -> FrozenDict:
        raise NotImplementedError
