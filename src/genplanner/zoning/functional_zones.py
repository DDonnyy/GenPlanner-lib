import math

from ._basic_terr_zones import *
from .abc_zone import Zone
from .territory_zones import TerritoryZone


class FunctionalZone(Zone):
    def __init__(self, zones_ratio: dict[TerritoryZone, float], name: str):
        self._name = name
        self._zones_ratio = dict(zones_ratio)

        self.validate()

        self._zones_ratio = self._normalize_ratio(self._zones_ratio)
        self._zones_keys = {z.name: z for z in self._zones_ratio}
        self._min_area = self._calc_min_area()

    @property
    def name(self) -> str:
        return self._name

    @property
    def min_area(self) -> float:
        return self._min_area

    @property
    def zones_ratio(self) -> dict[TerritoryZone, float]:
        return dict(self._zones_ratio)

    def validate(self) -> None:
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

    def _normalize_ratio(self, zones_ratio: dict[TerritoryZone, float]) -> dict[TerritoryZone, float]:
        total = float(sum(zones_ratio.values()))
        if total <= 0:
            raise ValueError("Total ratio must be > 0")
        return {z: float(v) / total for z, v in zones_ratio.items()}

    def _calc_min_area(self) -> float:
        return float(sum(z.min_area * r for z, r in self._zones_ratio.items()))

    def __str__(self) -> str:
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
