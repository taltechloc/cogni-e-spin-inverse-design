# optimizers/ga_optimizer.py
import numpy as np
from matplotlib import pyplot as plt

from id.optimizers.optimization_result import OptimizationResult
from id.optimizers.base_optimizer import BaseOptimizer

GAConfig = {
    "population_size": None,
    "generations": None,
    "crossover_rate": None,
    "mutation_rate": None,
    "mutation_scale": None,
    "early_stop_patience": None,
    "optimizer_type": "GA"
}


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Genetic Algorithm Optimizer for inverse design.
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.population_size = definition.get("population_size", 50)
        self.generations = definition.get("generations", 20)
        self.crossover_rate = definition.get("crossover_rate", 0.8)
        self.mutation_rate = definition.get("mutation_rate", 0.1)
        self.mutation_scale = definition.get("mutation_scale", 0.1)
        self.early_stop_patience = definition.get("early_stop_patience", 60)
        self.dim = len(boundaries)

        self.lower_bounds = np.array([b[0] for b in boundaries])
        self.upper_bounds = np.array([b[1] for b in boundaries])

    def optimize(self, target):
        # Initialize population
        population = np.random.uniform(self.lower_bounds, self.upper_bounds,
                                       size=(self.population_size, self.dim))

        def obj_fun(x):
            x = np.array(x).reshape(1, -1)
            pred = self.objective.model.predict(x)[0]
            return (pred - target) ** 2

        fitness = np.array([obj_fun(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        cost_history = []
        no_improvement_counter = 0
        top_candidates = [(best_fitness, best_solution.copy())]

        for gen in range(self.generations):

            # Tournament selection
            selected = self._tournament_selection(population, fitness)

            # Crossover
            offspring = self._crossover(selected)

            # Mutation
            self._mutate(offspring)

            # Evaluate offspring
            fitness_offspring = np.array([obj_fun(ind) for ind in offspring])

            # Combine and select the next generation
            combined_pop = np.vstack((population, offspring))
            combined_fit = np.hstack((fitness, fitness_offspring))
            best_indices = np.argsort(combined_fit)[:self.population_size]

            population = combined_pop[best_indices]
            fitness = combined_fit[best_indices]

            # Track current best
            current_best_index = np.argmin(fitness)
            current_best_fitness = fitness[current_best_index]
            current_best_solution = population[current_best_index].copy()

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best_solution.copy()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Update top candidates
            top_candidates += [(fitness[i], population[i].copy()) for i in range(self.population_size)]
            top_candidates = sorted(top_candidates, key=lambda t: t[0])[:5]

            cost_history.append(best_fitness)

            if no_improvement_counter >= self.early_stop_patience:
                break

        predicted = self.objective.model.predict(best_solution.reshape(1, -1))[0]
        top_positions = [p for c, p in top_candidates]
        plots_data = {
            "cost_history": self.plot_cost_history(
                cost_history,
                xlabel="Generation",
                ylabel="Squared Error",
                title="GA Convergence",
                label="Best Fitness per Generation"
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

    def _tournament_selection(self, pop, fit, k=3):
        selected = []
        pop_size = len(pop)
        for _ in range(self.population_size):
            aspirants_idx = np.random.choice(pop_size, k)
            aspirants_fit = fit[aspirants_idx]
            winner = aspirants_idx[np.argmin(aspirants_fit)]
            selected.append(pop[winner].copy())
        return np.array(selected)

    def _crossover(self, selected):
        offspring = []
        for i in range(0, self.population_size, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < self.population_size else selected[0]

            if np.random.rand() < self.crossover_rate:
                cross_pt = np.random.randint(1, self.dim)
                child1 = np.concatenate((parent1[:cross_pt], parent2[cross_pt:]))
                child2 = np.concatenate((parent2[:cross_pt], parent1[cross_pt:]))
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            offspring.append(child1)
            offspring.append(child2)
        return np.array(offspring)[:self.population_size]

    def _mutate(self, offspring):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.normal(0, self.mutation_scale, self.dim)
                offspring[i] += mutation
                offspring[i] = np.clip(offspring[i], self.lower_bounds, self.upper_bounds)
