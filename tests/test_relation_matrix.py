"""Checks for the adjacency rules passed to the Voronoi optimizer."""

import pandas as pd
import pytest

from genplanner import Relation, ZoneRelationMatrix
from genplanner.zones import BasicZone, TerritoryZone, TerritoryZoneKind


def test_dataframe_relations_round_trip_and_keep_zone_order():
    zones = [BasicZone("business"), BasicZone("residential"), BasicZone("park")]
    values = [["", "F", "N"], ["f", "", 0], [1, None, ""]]
    frame = pd.DataFrame(values, index=zones, columns=zones)

    matrix = ZoneRelationMatrix.from_dataframe(frame)

    assert matrix.zones == tuple(zones)
    assert matrix.get(zones[0], zones[1]) is Relation.FORBIDDEN
    assert matrix.get(zones[0], zones[2]) is Relation.NEIGHBOR
    assert matrix.get(zones[1], zones[2]) is Relation.NEUTRAL
    assert matrix.zone_forbidden() == [(zones[0], zones[1])]
    assert matrix.zone_neighbors() == [(zones[0], zones[2])]
    assert matrix.as_dataframe().loc[zones[1], zones[0]] is Relation.FORBIDDEN


def test_dataframe_rejects_asymmetric_relations():
    a, b = BasicZone("a"), BasicZone("b")
    frame = pd.DataFrame([["", "F"], ["N", ""]], index=[a, b], columns=[a, b])

    with pytest.raises(ValueError, match="not symmetric"):
        ZoneRelationMatrix.from_dataframe(frame)


def test_kind_forbidden_and_subset_preserve_constraints():
    residential = TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="residential")
    industrial = TerritoryZone(kind=TerritoryZoneKind.INDUSTRIAL, name="industrial")
    business = TerritoryZone(kind=TerritoryZoneKind.BUSINESS, name="business")
    matrix = ZoneRelationMatrix.from_kind_forbidden(
        [residential, industrial, business],
        {(TerritoryZoneKind.RESIDENTIAL, TerritoryZoneKind.INDUSTRIAL)},
    )

    assert matrix.get(residential, industrial) is Relation.FORBIDDEN
    assert matrix.get(industrial, residential) is Relation.FORBIDDEN
    assert matrix.get(residential, business) is Relation.NEUTRAL
    assert matrix.subset([business, industrial]).zones == (industrial, business)
    assert matrix.subset([residential, industrial]).zone_forbidden() == [(industrial, residential)]
    with pytest.raises(KeyError, match="not found"):
        matrix.subset([BasicZone("missing")], strict=True)
