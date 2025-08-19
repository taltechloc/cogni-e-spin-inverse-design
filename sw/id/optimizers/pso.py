# optimizers/pso.py
import numpy as np
from .base_optimizer import BaseOptimizer


class PSOOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimizer with inertia decay, velocity limits, and early stopping.
    """

    def __init__(self, objective, bounds, n_particles=50, n_iter=20,
                 w_max=0.9, w_min=0.4, c1=0.8, c2=0.9, max_velocity=0.2,
                 early_stop_patience=60):
        super().__init__(objective, bounds, n_iter)
        self.n_particles = n_particles
        self.dim = len(bounds)
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.max_velocity = max_velocity
        self.early_stop_patience = early_stop_patience

    def optimize(self, target):
        # Initialize particles and velocities
        lower_bounds = np.array([b[0] for b in self.bounds])
        upper_bounds = np.array([b[1] for b in self.bounds])
        param_ranges = upper_bounds - lower_bounds
        v_max = self.max_velocity * param_ranges

        particles = np.random.uniform(low=lower_bounds, high=upper_bounds,
                                      size=(self.n_particles, self.dim))
        velocities = np.zeros_like(particles)

        best_positions = particles.copy()
        best_costs = np.array([self.objective.evaluate(p, target) for p in particles])

        global_best_index = np.argmin(best_costs)
        global_best_position = best_positions[global_best_index].copy()
        global_best_cost = best_costs[global_best_index]

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
            velocities = np.clip(velocities, -v_max, v_max)

            particles += velocities
            particles = np.clip(particles, lower_bounds, upper_bounds)

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
        return global_best_position, predicted, cost_history
