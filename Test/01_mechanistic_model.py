import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os  # Add this to handle directories

def amr_dynamics(y, t, k1, k2, mutation_rate):
    """ODEs for antibiotic concentration and resistant bacteria growth."""
    """Simulate logistic bacterial growth with carrying capacity K, growth rate r."""
    Antibiotic, Sensitive, Resistant = y
    dAntibiotic_dt = -k1 * Antibiotic * Resistant
    dSensitive_dt = -mutation_rate * Sensitive + k2 * Sensitive * Antibiotic
    dResistant_dt = mutation_rate * Sensitive - k1 * Antibiotic * Resistant
    return [dAntibiotic_dt, dSensitive_dt, dResistant_dt]

def simulate_and_plot():
    # Create the 'plots' directory if it doesn't exist
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    # Initial conditions and parameters
    y0 = [10, 100, 5]  # [Antibiotic, Sensitive bacteria, Resistant bacteria]
    t = np.linspace(0, 48, 200)  # Simulate for 48 hours
    k1, k2, mutation_rate = 0.1, 0.02, 0.01

    # Solve ODEs
    solution = odeint(amr_dynamics, y0, t, args=(k1, k2, mutation_rate))

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, solution[:, 0], label='Antibiotic', linewidth=2)
    plt.plot(t, solution[:, 1], label='Sensitive Bacteria', linewidth=2)
    plt.plot(t, solution[:, 2], label='Resistant Bacteria', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Population / Concentration')
    plt.legend()
    plt.title('Mechanistic Model: Antibiotic Resistance Dynamics')
    plt.grid(True)

    # Save the plot to the 'plots' directory
    plt.savefig('./plots/mechanistic_model.png')
    plt.show()

if __name__ == '__main__':
    simulate_and_plot()
