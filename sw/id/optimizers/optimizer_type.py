from enum import Enum
from id.optimizers.pso import PSOOptimizer
from id.optimizers.random_search import RandomSearchOptimizer
from id.optimizers.knn_search import KNNSearch
from id.optimizers.ga import GAOptimizer
from id.optimizers.grid_search import GridSearchOptimizer



class OptimizerTypeError(Exception):
    pass


class OptimizerType(Enum):
    PSO = "PSO"
    RANDOM_SEARCH = "RANDOM_SEARCH"
    KNN_Search = "KNN_Search"
    GA = "GA"
    GRID_SEARCH = "GRID_SEARCH"


    def create(self, **kwargs):
        if self is OptimizerType.PSO:
            return PSOOptimizer(**kwargs)
        elif self is OptimizerType.RANDOM_SEARCH:
            return RandomSearchOptimizer(**kwargs)
        elif self is OptimizerType.KNN_Search:
            return KNN_Search(**kwargs)
        elif self is OptimizerType.GA:
            return GAOptimizer(**kwargs)
        elif self is OptimizerType.GRID_SEARCH:
            return GridSearchOptimizer(**kwargs)
        else:
            raise OptimizerTypeError("Unknown optimizer type: " + str(self))
