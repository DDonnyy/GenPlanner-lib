from dataclasses import dataclass

from genplanner.zones.abc_zone import Zone


@dataclass(frozen=True, slots=True)
class BasicZone(Zone):
    """
    A minimal zone implementation identified only by name.

    This is typically used for technical/intermediate zones (e.g., blocks)
    where only a stable string name is required.

    Attributes:
        name: Non-empty zone name.

    Raises:
        ValueError: If `name` is not a non-empty string.
    """

    name: str

    def __post_init__(self):
        """
        Validate dataclass fields after initialization.

        Raises:
            ValueError: If `name` is not a non-empty string.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Zone name must be non-empty string")

    def __str__(self) -> str:
        """
        Return a human-readable representation.

        Returns:
            str: String representation.
        """
        return f'Zone "{self.name}"'
