from genplanner.zones.territory_zones import TerritoryZoneKind

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