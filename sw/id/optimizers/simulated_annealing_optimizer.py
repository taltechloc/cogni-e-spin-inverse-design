# optimizers/sa_optimizer.py
import numpy as np
from matplotlib import pyplot as plt
from id.optimizers.optimization_result import OptimizationResult
from id.optimizers.base_optimizer import BaseOptimizer

SAConfig = {
    "initial_temperature": None,
    "cooling_rate": None,
    "iterations_per_temp": None,
    "step_size": None,
    "early_stop_patience": None,
    "optimizer_type": "SA"
}


class simulated_annealing_optimizer(BaseOptimizer):
    """
    Simulated Annealing Optimizer for inverse design.
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.initial_temperature = definition.get("initial_temperature", 1000.0)
        self.cooling_rate = definition.get("cooling_rate", 0.95)
        self.iterations_per_temp = definition.get("iterations_per_temp", 100)
        self.step_size = definition.get("step_size", 0.1)
        self.early_stop_patience = definition.get("early_stop_patience", 50)
        self.dim = len(boundaries)

        self.lower_bounds = np.array([b[0] for b in boundaries])
        self.upper_bounds = np.array([b[1] for b in boundaries])

    def optimize(self, target):
        def obj_fun(x):
            x = np.array(x).reshape(1, -1)
            pred = self.objective.model.predict(x)[0]
            return (pred - target) ** 2

        # Initialize current solution
        current_solution = np.random.uniform(self.lower_bounds, self.upper_bounds)
        current_cost = obj_fun(current_solution)

        best_solution = current_solution.copy()
        best_cost = current_cost

        temperature = self.initial_temperature
        cost_history = [best_cost]
        top_candidates = [(best_cost, best_solution.copy())]
        no_improvement_counter = 0

        while temperature > 1e-8:
            for _ in range(self.iterations_per_temp):
                # Generate neighbor
                neighbor = current_solution + np.random.normal(0, self.step_size, self.dim)
                neighbor = np.clip(neighbor, self.lower_bounds, self.upper_bounds)
                neighbor_cost = obj_fun(neighbor)

                # Acceptance criterion
                cost_diff = neighbor_cost - current_cost
                if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temperature):
                    current_solution = neighbor
                    current_cost = neighbor_cost

                # Update best solution
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                # Update top candidates
                top_candidates.append((current_cost, current_solution.copy()))
                top_candidates = sorted(top_candidates, key=lambda t: t[0])[:5]
                cost_history.append(best_cost)

                if no_improvement_counter >= self.early_stop_patience:
                    break

            if no_improvement_counter >= self.early_stop_patience:
                break

            # Cool down
            temperature *= self.cooling_rate

        predicted = self.objective.model.predict(best_solution.reshape(1, -1))[0]
        top_positions = [p for c, p in top_candidates]

        plots_data = {
            "cost_history": self.plot_cost_history(
                cost_history,
                xlabel="Iteration",
                ylabel="Squared Error",
                title="Simulated Annealing Convergence",
                label="Best Cost"
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
