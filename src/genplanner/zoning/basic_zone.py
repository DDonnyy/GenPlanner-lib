from dataclasses import dataclass

from genplanner.zoning.abc_zone import Zone


@dataclass(frozen=True, slots=True)
class BasicZone(Zone):
    name: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Zone name must be non-empty string")

    def __str__(self) -> str:
        return f'Zone "{self.name}"'
