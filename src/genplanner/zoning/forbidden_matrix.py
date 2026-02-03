from genplanner.zoning.terr_zones import TerritoryZoneKind

FORBIDDEN_NEIGHBORHOOD: set[tuple[TerritoryZoneKind, TerritoryZoneKind]] = {
    # Residential
    (TerritoryZoneKind.RESIDENTIAL, TerritoryZoneKind.SPECIAL),
    (TerritoryZoneKind.RESIDENTIAL, TerritoryZoneKind.INDUSTRIAL),
    (TerritoryZoneKind.RESIDENTIAL, TerritoryZoneKind.AGRICULTURE),
    (TerritoryZoneKind.RESIDENTIAL, TerritoryZoneKind.TRANSPORT),
    # Business
    (TerritoryZoneKind.BUSINESS, TerritoryZoneKind.SPECIAL),
    (TerritoryZoneKind.BUSINESS, TerritoryZoneKind.AGRICULTURE),
    # Special
    (TerritoryZoneKind.SPECIAL, TerritoryZoneKind.RESIDENTIAL),
    (TerritoryZoneKind.SPECIAL, TerritoryZoneKind.BUSINESS),
    # Industrial
    (TerritoryZoneKind.INDUSTRIAL, TerritoryZoneKind.RESIDENTIAL),
    # Agriculture
    (TerritoryZoneKind.AGRICULTURE, TerritoryZoneKind.RESIDENTIAL),
    (TerritoryZoneKind.AGRICULTURE, TerritoryZoneKind.BUSINESS),
    # Transport
    (TerritoryZoneKind.TRANSPORT, TerritoryZoneKind.RESIDENTIAL),
}


def make_symmetric(
    forbidden: set[tuple[TerritoryZoneKind, TerritoryZoneKind]],
) -> set[tuple[TerritoryZoneKind, TerritoryZoneKind]]:
    symmetric = set(forbidden)
    for a, b in forbidden:
        symmetric.add((b, a))
    return symmetric


FORBIDDEN_NEIGHBORHOOD = make_symmetric(FORBIDDEN_NEIGHBORHOOD)


def can_be_neighbors(a, b) -> bool:
    ka = a.kind if hasattr(a, "kind") else a
    kb = b.kind if hasattr(b, "kind") else b

    if ka == kb:
        return True

    return (ka, kb) not in FORBIDDEN_NEIGHBORHOOD
