# optimizers/pso.py
import numpy as np
from matplotlib import pyplot as plt

from id.optimizers.optimization_result import OptimizationResult
from id.optimizers.base_optimizer import BaseOptimizer


PSOConfig = {
    # Template configuration for PSOOptimizer
    # Actual values assignment in config.py
    "n_iter": None,              # int
    "n_particles": None,         # int
    "w_max": None,               # float
    "w_min": None,               # float
    "c1": None,                  # float
    "c2": None,                  # float
    "max_velocity": None,        # float
    "early_stop_patience": None, # int
    "optimizer_type": "PSO"      # str (default)
}

class PSOOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimizer with inertia decay, velocity limits, and early stopping.
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.n_iter = definition.get("n_iter")
        self.n_particles = definition.get("n_particles")
        self.dim = len(boundaries)
        self.w_max, self.w_min = definition.get("w_max"), definition.get("w_min")
        self.c1, self.c2 = definition.get("c1"), definition.get("c2")
        self.max_velocity = definition.get("max_velocity")
        self.early_stop_patience = definition.get("early_stop_patience")

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
        # Initialization
        particles, velocities, best_positions, best_costs, global_best_position, global_best_cost = \
            self.initialize_particles(target)

        cost_history = []
        swarm_diversity_history = []
        particle_trajectories = []
        no_improvement_counter = 0

        # Top 5 candidates
        top_candidates = [(c, p.copy()) for c, p in zip(best_costs, best_positions)]
        top_candidates = sorted(top_candidates, key=lambda x: x[0])[:5]

        for iteration in range(self.n_iter):
            # Update inertia
            w = self.w_max - (self.w_max - self.w_min) * (iteration / self.n_iter)
            r1, r2 = np.random.rand(self.n_particles, self.dim), np.random.rand(self.n_particles, self.dim)
            cognitive = self.c1 * r1 * (best_positions - particles)
            social = self.c2 * r2 * (global_best_position - particles)
            velocities = w * velocities + cognitive + social
            velocities = np.clip(velocities, -self.v_max, self.v_max)
            particles += velocities
            particles = np.clip(particles, self.lower_bounds, self.upper_bounds)

            costs = np.array([self.objective.evaluate(p, target) for p in particles])
            improved = costs < best_costs
            best_positions[improved] = particles[improved]
            best_costs[improved] = costs[improved]

            # Update global best
            current_best_index = np.argmin(best_costs)
            current_best_cost = best_costs[current_best_index]
            if current_best_cost < global_best_cost:
                global_best_cost = current_best_cost
                global_best_position = best_positions[current_best_index].copy()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Update top 5 candidates
            all_candidates = top_candidates + [(c, p.copy()) for c, p in zip(costs, particles)]
            top_candidates = sorted(all_candidates, key=lambda x: x[0])[:5]

            # Store histories
            cost_history.append(global_best_cost)
            swarm_diversity_history.append(self._compute_swarm_diversity(particles))
            particle_trajectories.append(particles.copy())

            if no_improvement_counter >= self.early_stop_patience:
                break

        top_positions = [p for c, p in top_candidates]
        predicted = self.objective.model.predict(global_best_position.reshape(1, -1))[0]

        # Generate plots and store in dictionary
        plots_data = self._generate_all_plots(cost_history, swarm_diversity_history, particle_trajectories)

        return OptimizationResult(
            best_candidates=global_best_position,
            best_prediction=predicted,
            cost_history=cost_history,
            top_candidates=top_positions,
            n_iterations=len(cost_history),
            plots_data=plots_data
        )

    # ----------------------
    # Plot generation methods
    # ----------------------
    def _generate_all_plots(self, cost_history, diversity_history, trajectories):
        """
        Generate all three plots and return in a dictionary.
        """
        plots = {}
        plots["cost_history"] = self._plot_cost_history(cost_history)
        plots["swarm_diversity"] = self._plot_swarm_diversity(diversity_history)
        return plots

    def _plot_cost_history(self, cost_history):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cost_history, label="Global Best Cost")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title("PSO Convergence")
        ax.legend()
        plt.close(fig)
        return fig

    def _plot_swarm_diversity(self, diversity_history):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(diversity_history)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Pairwise Distance")
        ax.set_title("Swarm Diversity")
        plt.close(fig)
        return fig

    def _compute_swarm_diversity(self, particles):
        """
        Compute mean pairwise distance between particles.
        """
        return np.mean(np.linalg.norm(particles[:, None, :] - particles[None, :, :], axis=2))
