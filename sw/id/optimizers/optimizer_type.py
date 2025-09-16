from enum import Enum
from id.optimizers.particle_swarm_optimizer import ParticleSwarmOptimizer
from id.optimizers.random_search_optimizer import RandomSearchOptimizer
from id.optimizers.genetic_algorithm_optimizer import GeneticAlgorithmOptimizer
from id.optimizers.grid_search_optimizer import GridSearchOptimizer
from id.optimizers.simulated_annealing_optimizer import simulated_annealing_optimizer
from id.optimizers.bayesian_optimizer import BayesianOptimizer
from id.optimizers.differential_evolution_optimizer import DifferentialEvolutionOptimizer



class OptimizerTypeError(Exception):
    pass


class OptimizerType(Enum):
    PSO = "PSO"
    RANDOM_SEARCH = "RANDOM_SEARCH"
    KNN_Search = "KNN_Search"
    GA = "GA"
    GRID_SEARCH = "GRID_SEARCH"
    SA = "SA"
    BO = "BO"
    DE = "DE"


    def create(self, **kwargs):
        if self is OptimizerType.PSO:
            return ParticleSwarmOptimizer(**kwargs)
        elif self is OptimizerType.RANDOM_SEARCH:
            return RandomSearchOptimizer(**kwargs)
        elif self is OptimizerType.GA:
            return GeneticAlgorithmOptimizer(**kwargs)
        elif self is OptimizerType.GRID_SEARCH:
            return GridSearchOptimizer(**kwargs)
        elif self is OptimizerType.SA:
            return simulated_annealing_optimizer(**kwargs)
        elif self is OptimizerType.BO:
            return BayesianOptimizer(**kwargs)
        elif self is OptimizerType.DE:
            return DifferentialEvolutionOptimizer(**kwargs)
        else:
            raise OptimizerTypeError("Unknown optimizer type: " + str(self))
