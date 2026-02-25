import math

from ._basic_terr_zones import *
from .abc_zone import Zone
from .territory_zones import TerritoryZone


def _normalize_ratio(zones_ratio: dict["TerritoryZone", float]) -> dict["TerritoryZone", float]:
    """
    Normalize ratios so that they sum to 1.0.

    Args:
        zones_ratio: Unnormalized ratios.

    Returns:
        dict[TerritoryZone, float]: Normalized ratios.

    Raises:
        ValueError: If the total ratio is not > 0.
    """
    total = float(sum(zones_ratio.values()))
    if total <= 0:
        raise ValueError("Total ratio must be > 0")
    return {z: float(v) / total for z, v in zones_ratio.items()}


class FunctionalZone(Zone):
    """
    A composite zone describing a functional program of territorial zones.

    `FunctionalZone` groups multiple `TerritoryZone` definitions with target
    area ratios. Ratios are validated and normalized to sum to 1.0.

    The `min_area` of a functional zone is computed as the weighted sum of
    member zones' `min_area` values (for `TerritoryZone` this corresponds to
    `min_block_area`), using the normalized ratios.

    Args:
        zones_ratio: Mapping of `TerritoryZone` to a positive (unnormalized)
            target ratio.
        name: Non-empty functional zone name.

    Raises:
        ValueError: If `name` is empty, `zones_ratio` is empty, ratios are non-positive,
            or the ratio sum is invalid.
        TypeError: If keys are not `TerritoryZone` or values are not finite numbers.
    """

    def __init__(self, zones_ratio: dict["TerritoryZone", float], name: str):
        self._name = name
        self._zones_ratio = dict(zones_ratio)

        self.validate()

        self._zones_ratio = _normalize_ratio(self._zones_ratio)
        self._zones_keys = {z.name: z for z in self._zones_ratio}
        self._min_area = self._calc_min_area()

    @property
    def name(self) -> str:
        """
        Functional zone name.

        Returns:
            str: Name of the functional zone.
        """
        return self._name

    @property
    def min_area(self) -> float:
        """
        Weighted minimum area constraint for the functional zone.

        Computed as `sum(zone.min_area * ratio)` using normalized ratios.

        Returns:
            float: Weighted minimum area.
        """
        return self._min_area

    @property
    def zones_ratio(self) -> dict["TerritoryZone", float]:
        """
        Normalized zone ratio mapping.

        Returns:
            dict[TerritoryZone, float]: A copy of the internal mapping where
            values sum to 1.0.
        """
        return dict(self._zones_ratio)

    def validate(self) -> None:
        """
        Validate functional zone invariants.

        Ensures:
          - name is a non-empty string,
          - zones_ratio is a non-empty dict,
          - keys are `TerritoryZone`,
          - values are finite numbers greater than 0.

        Raises:
            ValueError: If name is empty, zones_ratio is empty, or any ratio is <= 0.
            TypeError: If keys/values have incorrect types or ratios are not finite.
        """
        if not isinstance(self._name, str) or not self._name.strip():
            raise ValueError("FunctionalZone name must be non-empty string")

        if not isinstance(self._zones_ratio, dict) or not self._zones_ratio:
            raise ValueError("zones_ratio cannot be empty")

        for z, r in self._zones_ratio.items():
            if not isinstance(z, TerritoryZone):
                raise TypeError("All keys in zones_ratio must be TerritoryZone")
            if not isinstance(r, (int, float)) or not math.isfinite(float(r)):
                raise TypeError("All values in zones_ratio must be finite numbers")
            if float(r) <= 0:
                raise ValueError("All values in zones_ratio must be > 0")

    def _calc_min_area(self) -> float:
        """
        Compute the weighted minimum area for the functional zone.

        Returns:
            float: `sum(zone.min_area * ratio)` over normalized ratios.
        """
        return float(sum(z.min_area * r for z, r in self._zones_ratio.items()))

    def __str__(self) -> str:
        """
        Return a human-readable representation.

        Returns:
            str: String representation.
        """
        return f'Functional zone "{self.name}"'


basic_func_zone = FunctionalZone(
    {
        residential_terr: 0.25,
        industrial_terr: 0.12,
        business_terr: 0.08,
        recreation_terr: 0.3,
        transport_terr: 0.1,
        agriculture_terr: 0.03,
        special_terr: 0.02,
    },
    "basic",
)

residential_func_zone = FunctionalZone(
    {
        residential_terr: 0.5,
        business_terr: 0.1,
        recreation_terr: 0.1,
        transport_terr: 0.1,
        agriculture_terr: 0.05,
        special_terr: 0.05,
    },
    "residential territory",
)

industrial_func_zone = FunctionalZone(
    {
        industrial_terr: 0.5,
        business_terr: 0.1,
        recreation_terr: 0.05,
        transport_terr: 0.1,
        agriculture_terr: 0.05,
        special_terr: 0.05,
    },
    "industrial territory",
)
business_func_zone = FunctionalZone(
    {
        residential_terr: 0.1,
        business_terr: 0.5,
        recreation_terr: 0.1,
        transport_terr: 0.1,
        agriculture_terr: 0.05,
        special_terr: 0.05,
    },
    "business territory",
)
recreation_func_zone = FunctionalZone(
    {
        residential_terr: 0.2,
        business_terr: 0.1,
        recreation_terr: 0.5,
        transport_terr: 0.05,
        agriculture_terr: 0.1,
    },
    "recreation territory",
)
transport_func_zone = FunctionalZone(
    {
        industrial_terr: 0.1,
        business_terr: 0.05,
        recreation_terr: 0.05,
        transport_terr: 0.5,
        agriculture_terr: 0.05,
        special_terr: 0.05,
    },
    "transport territory",
)
agricalture_func_zone = FunctionalZone(
    {
        residential_terr: 0.1,
        industrial_terr: 0.1,
        business_terr: 0.05,
        recreation_terr: 0.1,
        transport_terr: 0.05,
        agriculture_terr: 0.5,
        special_terr: 0.05,
    },
    "agriculture territory",
)
special_func_zone = FunctionalZone(
    {
        residential_terr: 0.01,
        industrial_terr: 0.1,
        business_terr: 0.05,
        recreation_terr: 0.05,
        transport_terr: 0.05,
        agriculture_terr: 0.05,
        special_terr: 0.5,
    },
    "special territory",
)
