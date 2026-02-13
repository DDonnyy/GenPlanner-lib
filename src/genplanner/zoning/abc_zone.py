from abc import ABC, abstractmethod
from functools import total_ordering

@total_ordering
class BaseZone(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def min_area(self) -> float:
        ...

    @abstractmethod
    def validate(self) -> None:
        ...

    def __lt__(self, other):
        if not isinstance(other, BaseZone):
            return NotImplemented
        return (self.name, self.min_area) < (other.name, other.min_area)

    def __eq__(self, other):
        if not isinstance(other, BaseZone):
            return NotImplemented
        return (self.name, self.min_area) == (other.name, other.min_area)

    def __hash__(self):
        return hash((self.name, self.min_area))
