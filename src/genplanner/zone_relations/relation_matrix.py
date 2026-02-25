from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from genplanner.zones.abc_zone import Zone
from genplanner.zones.territory_zones import TerritoryZone, TerritoryZoneKind


class Relation(str, Enum):
    """Cell value in adjacency matrix."""

    NEUTRAL = "neutral"  # doesn't matter
    NEIGHBOR = "neighbor"  # must/can be neighbors
    FORBIDDEN = "forbidden"  # cannot be neighbors


RELATION_VALUE_MAP = {
    Relation.FORBIDDEN: Relation.FORBIDDEN,
    Relation.NEIGHBOR: Relation.NEIGHBOR,
    Relation.NEUTRAL: Relation.NEUTRAL,
    "forbidden": Relation.FORBIDDEN,
    "F": Relation.FORBIDDEN,
    "f": Relation.FORBIDDEN,
    "neighbor": Relation.NEIGHBOR,
    "N": Relation.NEIGHBOR,
    "n": Relation.NEIGHBOR,
    "neutral": Relation.NEUTRAL,
    "": Relation.NEUTRAL,
    None: Relation.NEUTRAL,
    0: Relation.NEUTRAL,
    1: Relation.NEIGHBOR,
    -1: Relation.FORBIDDEN,
}


@dataclass(frozen=True, slots=True)
class ZoneRelationMatrix:
    """
    Symmetric matrix of relations between zones.
    Access is O(1). Default relation is NEUTRAL.

    Symmetry invariant:
        _m[a][b] == _m[b][a] for all a,b
    """

    zones: tuple[Zone, ...]
    _m: dict[Zone, dict[Zone, Relation]]

    @classmethod
    def empty(cls, zones: Iterable[Zone], default: Relation = Relation.NEUTRAL) -> "ZoneRelationMatrix":
        zones_t = tuple(zones)
        cls._validate_unique(zones_t)
        m: dict[Zone, dict[Zone, Relation]] = {a: {b: default for b in zones_t} for a in zones_t}
        return cls(zones=zones_t, _m=m)

    @classmethod
    def from_pairs(
        cls,
        zones: Iterable[Zone],
        *,
        neighbors: Iterable[tuple[Zone, Zone]] = (),
        forbidden: Iterable[tuple[Zone, Zone]] = (),
        default: Relation = Relation.NEUTRAL,
    ) -> "ZoneRelationMatrix":
        """
        Build from explicit pairs. Input pairs may be non-symmetric —
        matrix will be symmetric internally anyway.
        """
        mat = cls.empty(zones, default=default)
        mat = mat.with_pairs(neighbors, Relation.NEIGHBOR)
        mat = mat.with_pairs(forbidden, Relation.FORBIDDEN)
        return mat

    @classmethod
    def from_kind_forbidden(
        cls,
        zones: Iterable[TerritoryZone],
        kind_forbidden: set[tuple[TerritoryZoneKind, TerritoryZoneKind]],
        *,
        default: Relation = Relation.NEUTRAL,
    ) -> "ZoneRelationMatrix":
        zones_t = tuple(zones)
        cls._validate_unique(zones_t)
        mat = cls.empty(zones_t, default=default)

        kind2zones: dict[TerritoryZoneKind, list[TerritoryZone]] = {}
        for z in zones_t:
            kind2zones.setdefault(z.kind, []).append(z)

        for ka, kb in kind_forbidden:
            for a in kind2zones.get(ka, ()):
                for b in kind2zones.get(kb, ()):
                    mat._m[a][b] = Relation.FORBIDDEN
                    mat._m[b][a] = Relation.FORBIDDEN

        return mat

    @classmethod
    def from_dataframe(
            cls,
            df: pd.DataFrame,
            *,
            value_map: Mapping[object, Relation] | None = None,
            default: Relation = Relation.NEUTRAL,
            require_symmetric: bool = True,
    ) -> "ZoneRelationMatrix":
        """
        df must be square.

        Requirements:
        - df.index contains Zone objects
        - df.columns contains Zone objects
        - set(index) == set(columns)
        - (optional) matrix must be symmetric (after mapping values)
        """
        if df.shape[0] != df.shape[1]:
            raise ValueError("DataFrame must be square (N x N)")

        if value_map is None:
            value_map = RELATION_VALUE_MAP

        # Validate index/columns are Zone
        idx_zones = tuple(df.index)
        col_zones = tuple(df.columns)

        if not idx_zones or not col_zones:
            raise ValueError("DataFrame index/columns cannot be empty")

        for z in idx_zones:
            if not isinstance(z, Zone):
                raise TypeError(f"df.index must contain Zone objects, got: {type(z)!r}")
        for z in col_zones:
            if not isinstance(z, Zone):
                raise TypeError(f"df.columns must contain Zone objects, got: {type(z)!r}")

        # Validate same zone set
        if set(idx_zones) != set(col_zones):
            missing_in_cols = set(idx_zones) - set(col_zones)
            missing_in_idx = set(col_zones) - set(idx_zones)
            raise ValueError(
                "df.index and df.columns must contain the same Zone set. "
                f"Missing in columns: {list(missing_in_cols)!r}; missing in index: {list(missing_in_idx)!r}"
            )

        zones_t = idx_zones  # keep df.index order
        cls._validate_unique(zones_t)

        mat = cls.empty(zones_t, default=default)

        if require_symmetric:
            for a in zones_t:
                for b in zones_t:
                    raw_ab = df.loc[a, b]
                    raw_ba = df.loc[b, a]
                    rel_ab = value_map.get(raw_ab, None)
                    rel_ba = value_map.get(raw_ba, None)
                    if rel_ab is None or rel_ba is None:
                        raise ValueError(f"Unmapped df values at ({a!r},{b!r}) or ({b!r},{a!r})")
                    if rel_ab != rel_ba:
                        raise ValueError(
                            f"DataFrame is not symmetric at ({a!r},{b!r}): {raw_ab!r}->{rel_ab} "
                            f"vs ({b!r},{a!r}): {raw_ba!r}->{rel_ba}"
                        )

        for a in zones_t:
            for b in zones_t:
                raw = df.loc[a, b]
                rel = value_map.get(raw, None)
                if rel is None:
                    raise ValueError(f"Unmapped df value at ({a!r},{b!r}): {raw!r}")
                mat._m[a][b] = rel
                mat._m[b][a] = rel

        return mat

    def with_pairs(
        self,
        pairs: Iterable[tuple[Zone, Zone]],
        rel: Relation,
    ) -> "ZoneRelationMatrix":
        """
        Applies pairs and enforces symmetry: (a,b) and (b,a) both set to rel.
        """
        m2 = {a: dict(row) for a, row in self._m.items()}
        zones_set = set(self.zones)

        for a, b in pairs:
            if a not in zones_set or b not in zones_set:
                raise KeyError(f"Unknown zone in pair: ({a}, {b})")
            m2[a][b] = rel
            m2[b][a] = rel

        return ZoneRelationMatrix(zones=self.zones, _m=m2)

    def get(self, a: Zone, b: Zone) -> Relation:
        return self._m[a][b]

    def is_forbidden(self, a: Zone, b: Zone) -> bool:
        return self.get(a, b) == Relation.FORBIDDEN

    def is_neighbor(self, a: Zone, b: Zone) -> bool:
        return self.get(a, b) == Relation.NEIGHBOR

    def zone_forbidden(self) -> list[tuple[Zone, Zone]]:
        return self._pairs_for(Relation.FORBIDDEN)

    def zone_neighbors(self) -> list[tuple[Zone, Zone]]:
        return self._pairs_for(Relation.NEIGHBOR)

    def _pairs_for(self, rel: Relation) -> list[tuple[Zone, Zone]]:
        out: list[tuple[Zone, Zone]] = []
        seen: set[tuple[Zone, Zone]] = set()

        for a in self.zones:
            row = self._m[a]
            for b in self.zones:
                if a is b:
                    continue
                if row[b] != rel:
                    continue
                key = (a, b) if (a.name, a.min_area) <= (b.name, b.min_area) else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
        return out

    def subset(
        self,
        zones: Iterable[Zone],
        *,
        strict: bool = False,
    ) -> "ZoneRelationMatrix":
        """
        Return a new matrix restricted to the provided zones (if present in this matrix).

        - If strict=False: silently ignores zones not found in current matrix.
        - If strict=True: raises KeyError if any provided zone is missing.
        """
        current = set(self.zones)

        requested = tuple(zones)
        if not requested:
            raise ValueError("subset zones cannot be empty")

        missing = [z for z in requested if z not in current]
        if missing and strict:
            raise KeyError(f"Zones not found in matrix: {missing!r}")

        req_set = set(z for z in requested if z in current)
        new_zones = tuple(z for z in self.zones if z in req_set)

        if not new_zones:
            raise ValueError("subset resulted in empty zone list")

        new_m: dict[Zone, dict[Zone, Relation]] = {a: {b: self._m[a][b] for b in new_zones} for a in new_zones}

        return ZoneRelationMatrix(zones=new_zones, _m=new_m)

    def as_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame with Zone objects as both index and columns.

        values:
          - "enum": Relation enum objects
          - "value": Relation.value strings
        """

        data: list[list[object]] = []
        for a in self.zones:
            row: list[object] = []
            for b in self.zones:
                rel = self._m[a][b]
                row.append(rel)
            data.append(row)

        return pd.DataFrame(data=data, index=list(self.zones), columns=list(self.zones))

    @staticmethod
    def _validate_unique(zones: tuple[Zone, ...]) -> None:
        if not zones:
            raise ValueError("zones cannot be empty")
        if len(set(zones)) != len(zones):
            raise ValueError("zones must be unique (duplicates by Zone equality/hash found)")
        for z in zones:
            if not isinstance(z, Zone):
                raise TypeError(f"All zones must be Zone instances, got: {type(z)!r}")
            z.validate()

    def __str__(self) -> str:
        return self.to_pretty_string()

    def __repr__(self) -> str:
        return f"ZoneRelationMatrix(n={len(self.zones)})\n{self.to_pretty_string()}"

    def to_pretty_string(
        self,
        *,
        name_width: int = 5,
        cell_width: int = 6,
        neutral: str = "·",
        neighbor: str = "✓",
        forbidden: str = "✗",
        show_diagonal: bool = False,
    ) -> str:
        if name_width < 3:
            raise ValueError("name_width must be >= 3")

        sym = {
            Relation.NEUTRAL: neutral,
            Relation.NEIGHBOR: neighbor,
            Relation.FORBIDDEN: forbidden,
        }

        zones = self.zones
        labels = self._unique_short_labels(zones, width=name_width)

        left_w = max(len(lbl) for lbl in labels.values())
        left_w = max(left_w, name_width)

        cw = max(cell_width, 3)

        header = " " * (left_w + 1)
        for z in zones:
            header += labels[z].center(cw)
        lines = [header]

        for a in zones:
            row = labels[a].ljust(left_w) + " "
            for b in zones:
                if (a is b) and not show_diagonal:
                    ch = neutral
                else:
                    ch = sym[self._m[a][b]]
                row += str(ch).center(cw)
            lines.append(row)

        return "\n".join(lines)

    @staticmethod
    def _unique_short_labels(zones: tuple[Zone, ...], *, width: int) -> dict[Zone, str]:
        """
        Build short labels based on zone.name trimmed to width.
        If collisions happen, add suffix ~2, ~3 ... (still keeping <= width).
        """

        def trim(s: str) -> str:
            s = s.strip()
            if len(s) <= width:
                return s
            # keep last char as ellipsis marker
            if width <= 1:
                return s[:width]
            return s[: width - 1]

        base: dict[Zone, str] = {z: trim(getattr(z, "name", str(z))) for z in zones}
        used: dict[str, int] = {}
        out: dict[Zone, str] = {}

        for z in zones:
            b = base[z]
            k = used.get(b, 0) + 1
            used[b] = k
            if k == 1:
                out[z] = b
            else:
                suffix = f"~{k}"
                # ensure label fits width
                keep = max(width - len(suffix), 1)
                raw = b
                if len(raw) > keep:
                    raw = raw[:keep]
                out[z] = raw + suffix

        return out
