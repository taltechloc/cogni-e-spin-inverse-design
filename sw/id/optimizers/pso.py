# optimizers/pso.py
import numpy as np
from dataclasses import dataclass

from id.optimizers._optimization_result import _OptimizationResult
from id.optimizers.base_optimizer import BaseOptimizer


@dataclass
class PSOConfig:
    """
    Template configuration class for PSOOptimizer.
    Actual values assignment in config.py.
    """
    n_iter: int
    n_particles: int
    w_max: float
    w_min: float
    c1: float
    c2: float
    max_velocity: float
    early_stop_patience: int
    optimizer_type: str = "PSO"


class PSOOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimizer with inertia decay, velocity limits, and early stopping.
    """

    def __init__(self, config, objective, boundaries):
        super().__init__(objective, boundaries)
        self.n_iter = config.n_iter
        self.n_particles = config.n_particles
        self.dim = len(boundaries)
        self.w_max, self.w_min = config.w_max, config.w_min
        self.c1, self.c2 = config.c1, config.c2
        self.max_velocity = config.max_velocity
        self.early_stop_patience = config.early_stop_patience

        # Derived values
        self.lower_bounds = np.array([b[0] for b in self.boundaries])
        self.upper_bounds = np.array([b[1] for b in self.boundaries])
        self.param_ranges = self.upper_bounds - self.lower_bounds
        self.v_max = self.max_velocity * self.param_ranges

    def initialize_particles(self, target):
        """
        Initializes particles, velocities, best positions, and costs.
        """
        particles = np.random.uniform(low=self.lower_bounds, high=self.upper_bounds,
                                      size=(self.n_particles, self.dim))
        velocities = np.zeros_like(particles)

        best_positions = particles.copy()
        best_costs = np.array([self.objective.evaluate(p, target) for p in particles])

        global_best_index = np.argmin(best_costs)
        global_best_position = best_positions[global_best_index].copy()
        global_best_cost = best_costs[global_best_index]

        return particles, velocities, best_positions, best_costs, global_best_position, global_best_cost

    def optimize(self, target):
        # Initialization step
        particles, velocities, best_positions, best_costs, global_best_position, global_best_cost = \
            self.initialize_particles(target)

        cost_history = []
        no_improvement_counter = 0

        for iteration in range(self.n_iter):
            # Inertia weight decay
            w = self.w_max - (self.w_max - self.w_min) * (iteration / self.n_iter)

            # Update velocities and positions
            r1, r2 = np.random.rand(self.n_particles, self.dim), np.random.rand(self.n_particles, self.dim)
            cognitive = self.c1 * r1 * (best_positions - particles)
            social = self.c2 * r2 * (global_best_position - particles)
            velocities = w * velocities + cognitive + social
            velocities = np.clip(velocities, -self.v_max, self.v_max)

            particles += velocities
            particles = np.clip(particles, self.lower_bounds, self.upper_bounds)

            # Evaluate
            costs = np.array([self.objective.evaluate(p, target) for p in particles])

            improved = costs < best_costs
            best_positions[improved] = particles[improved]
            best_costs[improved] = costs[improved]

            current_best_index = np.argmin(best_costs)
            current_best_cost = best_costs[current_best_index]

            if current_best_cost < global_best_cost:
                global_best_cost = current_best_cost
                global_best_position = best_positions[current_best_index].copy()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            cost_history.append(global_best_cost)

            if no_improvement_counter >= self.early_stop_patience:
                break

        predicted = self.objective.model.predict(global_best_position.reshape(1, -1))[0]
        return _OptimizationResult(
            best_candidates=global_best_position,
            best_prediction=predicted,
            cost_history=cost_history,
            n_iterations=len(cost_history)
        )
