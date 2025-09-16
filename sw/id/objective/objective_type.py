from enum import Enum
from id.objective.surrogate_objective import SurrogateObjective


class ObjectiveTypeError(Exception):
    pass


class ObjectiveType(Enum):
    SURROGATE = "SurrogateObjective"

    def create(self, model, **kwargs):
        if self is ObjectiveType.SURROGATE:
            return SurrogateObjective(model, **kwargs)
        else:
            raise ObjectiveTypeError("Unknown objective type: " + str(self))
