from abc import ABC, abstractmethod
from functools import total_ordering


@total_ordering
class Zone(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    def validate(self) -> None:
        pass

    @property
    def min_area(self) -> float:
        return 1.0

    def __lt__(self, other: "Zone"):
        if not isinstance(other, Zone):
            return NotImplemented
        return (self.name, self.min_area) < (other.name, other.min_area)

    def __eq__(self, other: "Zone"):
        if not isinstance(other, Zone):
            return NotImplemented
        return (self.name, self.min_area) == (other.name, other.min_area)

    def __hash__(self):
        return hash((self.name, self.min_area))
