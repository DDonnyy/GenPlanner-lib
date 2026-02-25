from abc import ABC, abstractmethod
from functools import total_ordering


@total_ordering
class Zone(ABC):
    """
    Abstract base class for all zone types.

    A `Zone` is a lightweight, comparable entity used across the pipeline to
    describe named zoning concepts (e.g., territorial zone kinds, functional
    zone groups, or temporary technical zones). Zones are orderable and hashable
    based on `(name, min_area)` so they can be used as dictionary keys and
    sorted deterministically.

    Subclasses must provide a `name` and may override validation and area
    constraints.

    Notes:
        - Ordering and equality are defined by `(name, min_area)`, not identity.
        - `min_area` is a generic minimum area constraint. For territorial zones
          it typically represents the minimum block area.

    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable zone name.

        Returns:
            str: Zone name.
        """
        ...

    def validate(self) -> None:
        """
        Validate the zone configuration.

        Subclasses should override this method and raise an exception if the
        zone is invalid.

        Raises:
            Exception: Implementations should raise `ValueError`/`TypeError`
            (or custom errors) when invariants are violated.
        """
        pass

    @property
    def min_area(self) -> float:
        """
        Minimum area constraint associated with the zone.

        Returns:
            float: Minimum allowed area. Defaults to 1.0.
        """
        return 1.0

    def __lt__(self, other: "Zone"):
        """
        Compare zones for sorting.

        Ordering is based on `(name, min_area)`.

        Args:
            other: Another zone instance.

        Returns:
            bool: True if `self` is ordered before `other`.

        """
        if not isinstance(other, Zone):
            return NotImplemented
        return (self.name, self.min_area) < (other.name, other.min_area)

    def __eq__(self, other: "Zone"):
        """
        Compare zones for equality.

        Equality is based on `(name, min_area)`.

        Args:
            other: Another zone instance.

        Returns:
            bool: True if both zones have equal `name` and `min_area`.

        """
        if not isinstance(other, Zone):
            return NotImplemented
        return (self.name, self.min_area) == (other.name, other.min_area)

    def __hash__(self):
        """
        Hash for use in sets and as dictionary keys.

        Returns:
            int: Hash of `(name, min_area)`.
        """
        return hash((self.name, self.min_area))
