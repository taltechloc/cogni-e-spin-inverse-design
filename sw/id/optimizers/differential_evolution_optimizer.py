import numpy as np
from matplotlib import pyplot as plt
from id.optimizers.optimization_result import OptimizationResult
from id.optimizers.base_optimizer import BaseOptimizer

DEConfig = {
    "population_size": None,
    "generations": None,
    "crossover_rate": None,
    "mutation_factor": None,
    "strategy": None,
    "early_stop_patience": None,
    "optimizer_type": "DE"
}


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """
    Differential Evolution Optimizer for inverse design.
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.population_size = definition.get("population_size", 50)
        self.generations = definition.get("generations", 100)
        self.crossover_rate = definition.get("crossover_rate", 0.7)
        self.mutation_factor = definition.get("mutation_factor", 0.8)
        self.strategy = definition.get("strategy", "rand1bin")
        self.early_stop_patience = definition.get("early_stop_patience", 50)
        self.dim = len(boundaries)

        self.lower_bounds = np.array([b[0] for b in boundaries])
        self.upper_bounds = np.array([b[1] for b in boundaries])

    def optimize(self, target):
        def obj_fun(x):
            x = np.array(x).reshape(1, -1)
            pred = self.objective.model.predict(x)[0]
            return (pred - target) ** 2

        # Initialize population
        population = np.random.uniform(self.lower_bounds, self.upper_bounds,
                                       size=(self.population_size, self.dim))
        fitness = np.array([obj_fun(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        cost_history = [best_fitness]
        top_candidates = [(best_fitness, best_solution.copy())]
        no_improvement_counter = 0

        for gen in range(self.generations):
            new_population = population.copy()
            new_fitness = fitness.copy()

            for i in range(self.population_size):
                # Mutation
                if self.strategy == "rand1bin":
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = a + self.mutation_factor * (b - c)
                else:
                    # Add more strategies if needed
                    mutant = population[i] + self.mutation_factor * (best_solution - population[i])

                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = obj_fun(trial)
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()
                        no_improvement_counter = 0
                    else:
                        no_improvement_counter += 1

            population = new_population
            fitness = new_fitness
            cost_history.append(best_fitness)

            # Update top candidates
            top_candidates += [(fitness[i], population[i].copy()) for i in range(self.population_size)]
            top_candidates = sorted(top_candidates, key=lambda t: t[0])[:5]

            if no_improvement_counter >= self.early_stop_patience:
                break

        predicted = self.objective.model.predict(best_solution.reshape(1, -1))[0]
        top_positions = [p for c, p in top_candidates]
        plots_data = {
            "cost_history": self.plot_cost_history(
                cost_history,
                xlabel="Generation",
                ylabel="Squared Error",
                title="Differential Evolution Convergence",
                label="Best Fitness"
            )
        }
        return OptimizationResult(
            best_candidates=best_solution,
            best_prediction=predicted,
            cost_history=cost_history,
            top_candidates=top_positions,
            n_iterations=len(cost_history),
            plots_data=plots_data
        )
