class SplitPolygonValidationError(ValueError):
    """Raised when _split_polygon input arguments fail fast validation."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


class GenPlannerRuntimeError(RuntimeError):
    pass


class GenPlannerInitError(Exception):
    """Raised when GenPlanner initialization fails."""

    pass


class RelationMatrixError(Exception):
    """Raised when there is an error in ZoneRelationMatrix construction or usage."""

    pass


class GenPlannerArgumentError(ValueError):
    """Raised when invalid arguments are passed to GenPlanner methods."""

    pass


class FixPointsOutsideTerritoryError(GenPlannerArgumentError):
    """Raised when fixed points are outside the territory polygon."""

    pass


class GenplannerInfeasibleMultiFeatureError(GenPlannerRuntimeError):
    pass
