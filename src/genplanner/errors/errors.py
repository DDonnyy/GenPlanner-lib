class GenPlannerBaseError(Exception):
    def __init__(self, message: str, run_name: str | None = None):
        self.run_name = run_name
        self.message = message
        super().__init__(message, run_name)

    def __str__(self):
        if self.run_name:
            return f"[{self.run_name}] {self.message}"
        return self.message


class SplitPolygonValidationError(GenPlannerBaseError):
    def __init__(self, field: str, message: str, run_name: str | None = None):
        self.field = field
        full_message = f"{field}: {message}"
        super().__init__(full_message, run_name)


class GenPlannerInitError(GenPlannerBaseError):
    """Raised when GenPlanner initialization fails."""


class RelationMatrixError(GenPlannerBaseError):
    """Raised when there is an error in relation matrix construction or usage."""


class GenPlannerArgumentError(GenPlannerBaseError):
    """Raised when invalid arguments are passed to GenPlanner methods."""


class FixPointsOutsideTerritoryError(GenPlannerArgumentError):
    """Raised when fixed points are outside the territory polygon."""


class GenplannerInfeasibleMultiFeatureError(GenPlannerBaseError):
    """Raised when multi-feature setup is infeasible."""
