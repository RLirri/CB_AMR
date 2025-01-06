import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Bacteria:
    """A class representing a single bacterium."""
    def __init__(self, resistance=False):
        self.resistance = resistance

def simulate_abm(
    num_bacteria,
    steps,
    resistance_ratio=0.5,
    mutation_rate=0.01,
    loss_rate=0.01,
    growth_rate=0.1,
    carrying_capacity=1000,
    return_snapshots=False
):
    resistant = int(num_bacteria * resistance_ratio)
    non_resistant = num_bacteria - resistant
    history = []
    snapshots = []

    for step in range(steps):
        # Probabilistic mutations (Non-Resistant to Resistant)
        mutations = sum(1 for _ in range(non_resistant) if random.random() < mutation_rate)
        resistant += mutations
        non_resistant -= mutations

        # Probabilistic loss of resistance (Resistant to Non-Resistant)
        losses = sum(1 for _ in range(resistant) if random.random() < loss_rate)
        resistant -= losses
        non_resistant += losses

        # Population growth (Resistant and Non-Resistant grow proportionally)
        total_population = resistant + non_resistant
        new_bacteria = int(total_population * growth_rate * (1 - total_population / carrying_capacity))
        if total_population > 0:
            resistant += int(new_bacteria * (resistant / total_population))
            non_resistant += int(new_bacteria * (non_resistant / total_population))

        # Ensure populations are non-negative and within carrying capacity
        resistant = max(min(resistant, carrying_capacity), 0)
        non_resistant = max(min(non_resistant, carrying_capacity), 0)

        # Record the proportion of Resistant bacteria
        total_population = resistant + non_resistant
        history.append(resistant / total_population if total_population > 0 else 0)

        if return_snapshots:
            snapshots.append((resistant, non_resistant))

    if return_snapshots:
        return history, snapshots
    return history
