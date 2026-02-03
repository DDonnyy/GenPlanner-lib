from dataclasses import dataclass
from enum import Enum
from functools import total_ordering


class TerritoryZoneKind(str, Enum):
    RESIDENTIAL = "residential"
    INDUSTRIAL = "industrial"
    BUSINESS = "business"
    RECREATION = "recreation"
    TRANSPORT = "transport"
    AGRICULTURE = "agriculture"
    SPECIAL = "special"


minimum_block_area = 25000  # m^2


@total_ordering
@dataclass(frozen=True, slots=True)
class TerritoryZone:

    kind: TerritoryZoneKind
    name: str
    min_block_area: float = minimum_block_area

    def __post_init__(self):
        if not isinstance(self.kind, TerritoryZoneKind):
            try:
                object.__setattr__(self, "kind", TerritoryZoneKind(str(self.kind)))
            except Exception as e:
                raise ValueError(
                    f"Invalid TerritoryZone kind: {self.kind!r}. " f"Allowed: {[k.value for k in TerritoryZoneKind]}"
                ) from e

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TerritoryZone name must be a non-empty string")

        if self.min_block_area <= 0:
            raise ValueError("min_block_area must be > 0")

    def __str__(self) -> str:
        return f'Territory zone "{self.name}" ({self.kind.value})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__hash__() == other.__hash__()
        else:
            return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, TerritoryZone):
            return NotImplemented
        return (self.kind.value, self.name, self.min_block_area) < (
            other.kind.value,
            other.name,
            other.min_block_area,
        )


residential_terr = TerritoryZone(
    kind=TerritoryZoneKind.RESIDENTIAL,
    name="residential",
    min_block_area=minimum_block_area,
)

industrial_terr = TerritoryZone(
    kind=TerritoryZoneKind.INDUSTRIAL,
    name="industrial",
    min_block_area=minimum_block_area * 8,
)

business_terr = TerritoryZone(
    kind=TerritoryZoneKind.BUSINESS,
    name="business",
    min_block_area=minimum_block_area * 2,
)

recreation_terr = TerritoryZone(
    kind=TerritoryZoneKind.RECREATION,
    name="recreation",
    min_block_area=minimum_block_area * 4,
)

transport_terr = TerritoryZone(
    kind=TerritoryZoneKind.TRANSPORT,
    name="transport",
    min_block_area=minimum_block_area * 8,
)

agriculture_terr = TerritoryZone(
    kind=TerritoryZoneKind.AGRICULTURE,
    name="agriculture",
    min_block_area=minimum_block_area * 12,
)

special_terr = TerritoryZone(
    kind=TerritoryZoneKind.SPECIAL,
    name="special",
    min_block_area=minimum_block_area * 2,
)
