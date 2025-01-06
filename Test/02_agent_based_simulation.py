import numpy as np
import matplotlib.pyplot as plt

class Bacteria:
    def __init__(self, resistant=False):
        self.resistant = resistant

    def mutate(self, mutation_rate):
        if not self.resistant and np.random.rand() < mutation_rate:
            self.resistant = True

def simulate_bacterial_population(initial_population, mutation_rate, hours):
    bacteria = [Bacteria() for _ in range(initial_population)]
    resistant_counts = []

    for _ in range(hours):
        for bacterium in bacteria:
            bacterium.mutate(mutation_rate)
        resistant_count = sum([1 for b in bacteria if b.resistant])
        resistant_counts.append(resistant_count)

    # Plotting the simulation results
    plt.figure(figsize=(10, 6))
    plt.plot(range(hours), resistant_counts, label='Resistant Bacteria', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Resistant Bacteria Count')
    plt.title('Agent-Based Simulation: Mutation and Resistance Evolution')
    plt.grid(True)
    plt.savefig('./plots/agent_based_simulation.png')
    plt.show()

if __name__ == '__main__':
    simulate_bacterial_population(100, mutation_rate=0.01, hours=48)
