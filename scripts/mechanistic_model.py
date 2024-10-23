import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Reference: https://web.pdx.edu/~rueterj/courses/ESM221/Lect4-logistic.pptx
def logistic_growth(N, t, r, K):
    """Logistic growth equation: dN/dt = r * N * (1 - N / K)"""
    dNdt = r * N * (1 - N / K)
    return dNdt

def extract_growth_params(y):
    """Calculate growth parameters (r and K) based on dataset resistance levels."""
    resistance_ratio = y.sum().mean() / len(y)  # Average proportion of resistant samples
    r = 0.05 + resistance_ratio * 0.1  # Adjust growth rate based on resistance ratio
    K = 500 + int(resistance_ratio * 1000)  # Adjust carrying capacity dynamically
    return r, K

def simulate_logistic_growth(y_train):
    """
    Simulate logistic growth with parameters derived from the training dataset.
    This dynamically adjusts growth rate and carrying capacity.
    """
    r, K = extract_growth_params(y_train)  # Get dynamic growth parameters
    N0 = 1  # Initial population
    t = np.linspace(0, 100, 100)  # Time points

    # Solve the differential equation using odeint
    solution = odeint(logistic_growth, N0, t, args=(r, K))

    # Plot the logistic growth curve
    plt.plot(t, solution, label=f"Logistic Growth (r={r:.2f}, K={K})")
    plt.title("Mechanistic Modeling: Logistic Growth of Resistant Bacteria")
    plt.xlabel("Time")
    plt.ylabel("Population Size")
    plt.legend()
    plt.show()

    return solution
