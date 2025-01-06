import random
import matplotlib.pyplot as plt

class Bacteria:
    """A class representing a single bacterium."""
    def __init__(self, resistance=False):
        self.resistance = resistance

def simulate_abm(num_bacteria=100, steps=50, y_train=None):
    """
    Simulate the evolution of antibiotic resistance using an agent-based model (ABM).
    The probability of resistance is initialized based on the training dataset.
    """
    # Calculate resistance probability from the dataset
    prob_resistance = y_train.sum().mean() / len(y_train) if y_train is not None else 0.1

    # Initialize the bacteria population.
    # A population of Bacteria objects is created, with each bacterium having a
    # random chance of being resistant according to the calculated probability.
    population = [Bacteria(resistance=random.random() < prob_resistance) for _ in range(num_bacteria)]
    history = []  # Store the proportion of resistant bacteria over time

    # Simulate over time steps
    for step in range(steps):
        resistant_count = sum(b.resistance for b in population)
        history.append(resistant_count / num_bacteria)

        # Random mutations: Resistance may switch.
        # Each bacterium has a 1% chance of flipping its resistance status (mutation).
        for b in population:
            if random.random() < 0.01:
                b.resistance = not b.resistance

    # Plot the evolution of resistance
    plt.plot(history, label="ABM: Proportion of Resistant Bacteria", linestyle='--')
    plt.title("Agent-Based Modeling: Resistance Evolution")
    plt.xlabel("Time Steps")
    plt.ylabel("Proportion of Resistant Bacteria")
    plt.legend()
    plt.show()

    return history
