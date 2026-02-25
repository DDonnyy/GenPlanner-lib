from dataclasses import dataclass
from enum import Enum

from genplanner.zones.abc_zone import Zone
from genplanner import config


class TerritoryZoneKind(str, Enum):
    """
    Enumeration of supported territorial zone kinds.

    Values are stable string identifiers used in outputs and configuration.
    """

    RESIDENTIAL = "residential"
    INDUSTRIAL = "industrial"
    BUSINESS = "business"
    RECREATION = "recreation"
    TRANSPORT = "transport"
    AGRICULTURE = "agriculture"
    SPECIAL = "special"


minimum_block_area = config.minimum_block_area


@dataclass(frozen=True, slots=True)
class TerritoryZone(Zone):
    """
    A concrete territorial zone definition.

    `TerritoryZone` represents a specific zone within a territory plan. It has:
      - a `kind` (from :class:`TerritoryZoneKind`),
      - a human-readable `name`,
      - a minimum block area constraint (`min_block_area`).

    The zone's `min_area` is defined as `min_block_area`, making it compatible
    with generic area constraints used elsewhere in the pipeline.

    Attributes:
        kind: Zone kind (Enum or a value convertible to it).
        name: Non-empty zone name.
        min_block_area: Minimum allowed area for generated blocks for this zone.
            Must be > 0.

    Raises:
        ValueError: If `kind` is invalid, `name` is empty, or `min_block_area` <= 0.
    """

    kind: "TerritoryZoneKind"
    name: str
    min_block_area: float = minimum_block_area

    @property
    def min_area(self) -> float:
        """
        Minimum area constraint for this zone.

        Returns:
            float: Minimum block area.
        """
        return self.min_block_area

    def __post_init__(self):
        """
        Validate and normalize fields after initialization.

        Raises:
            ValueError: If any field violates validation rules.
        """
        self.validate()

    def validate(self) -> None:
        """
        Validate and normalize zone fields.

        Ensures:
          - `kind` is a `TerritoryZoneKind` (attempts coercion from string/other),
          - `name` is a non-empty string,
          - `min_block_area` is a positive number.

        Raises:
            ValueError: If `kind` cannot be coerced, `name` is empty,
                or `min_block_area` <= 0.
        """
        if not isinstance(self.kind, TerritoryZoneKind):
            try:
                object.__setattr__(self, "kind", TerritoryZoneKind(str(self.kind)))
            except Exception as e:
                raise ValueError(
                    f"Invalid TerritoryZone kind: {self.kind!r}. " f"Allowed: {[k.value for k in TerritoryZoneKind]}"
                ) from e

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Zone name must be non-empty string")

        if not isinstance(self.min_block_area, (int, float)) or self.min_block_area <= 0:
            raise ValueError("min_block_area must be > 0")

    def __str__(self) -> str:
        """
        Return a human-readable representation.

        Returns:
            str: String representation.
        """
        return f'Territory zone "{self.name}" ({self.kind.value})'


residential_terr = TerritoryZone(
    kind=TerritoryZoneKind.RESIDENTIAL,
    name="residential",
    min_block_area=minimum_block_area,
)

industrial_terr = TerritoryZone(
    kind=TerritoryZoneKind.INDUSTRIAL,
    name="industrial",
    min_block_area=minimum_block_area * 10,
)

business_terr = TerritoryZone(
    kind=TerritoryZoneKind.BUSINESS,
    name="business",
    min_block_area=minimum_block_area * 2,
)

recreation_terr = TerritoryZone(
    kind=TerritoryZoneKind.RECREATION,
    name="recreation",
    min_block_area=minimum_block_area * 8,
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
    min_block_area=minimum_block_area * 4,
)
