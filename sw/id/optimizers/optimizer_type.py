from enum import Enum
from id.optimizers.pso import PSOOptimizer


class OptimizerTypeError(Exception):
    pass


class OptimizerType(Enum):
    PSO = "PSOOptimizer"

    def create(self, **kwargs):
        if self is OptimizerType.PSO:
            return PSOOptimizer(**kwargs)
        else:
            raise OptimizerTypeError("Unknown optimizer type: " + str(self))
