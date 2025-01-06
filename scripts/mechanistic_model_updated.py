import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Reference: https://web.pdx.edu/~rueterj/courses/ESM221/Lect4-logistic.pptx
def logistic_growth_with_resistance(N, t, r_resistant, r_non_resistant, K, loss_probability, mutation_rate):
    """
    Logistic growth model that accounts for resistant and non-resistant populations.
    Includes resistance loss and mutation.
    """
    N_resistant, N_non_resistant = N
    total_population = N_resistant + N_non_resistant

    # Logistic growth dynamics
    dN_resistant = r_resistant * N_resistant * (1 - total_population / K)  # Logistic growth for resistant
    dN_non_resistant = r_non_resistant * N_non_resistant * (1 - total_population / K)  # Logistic growth for non-resistant

    # Resistance dynamics
    dN_resistant -= loss_probability * N_resistant  # Resistance loss
    dN_resistant += mutation_rate * N_non_resistant  # Mutation to resistance

    # Non-resistance dynamics
    dN_non_resistant += loss_probability * N_resistant  # Gain from resistance loss
    dN_non_resistant -= mutation_rate * N_non_resistant  # Loss due to mutation

    return [dN_resistant, dN_non_resistant]


def extract_growth_params(y):
    """Calculate growth parameters (r and K) based on dataset resistance levels."""
    resistance_ratio = y.sum().mean() / len(y)  # Average proportion of resistant samples
    r = 0.05 + resistance_ratio * 0.1  # Adjust growth rate based on resistance ratio
    K = 500 + int(resistance_ratio * 1000)  # Adjust carrying capacity dynamically
    return r, K

def simulate_logistic_growth_with_resistance(
    y_train, initial_population, growth_rate, carrying_capacity, duration, steps, loss_probability, mutation_rate
):
    """
    Simulate logistic growth for resistant and non-resistant populations.
    Uses the logistic_growth_with_resistance model for dynamics.
    """
    # Extract parameters from dataset
    r, K = extract_growth_params(y_train)

    # Override parameters with user inputs
    r_resistant = growth_rate if growth_rate > 0 else r  # Use user-provided growth rate for resistant population
    r_non_resistant = r_resistant * 0.8  # Assume non-resistant grows slightly slower
    K = carrying_capacity if carrying_capacity >= initial_population else K

    # Initial populations
    N_resistant = int(initial_population * 0.5)  # Start with half resistant
    N_non_resistant = initial_population - N_resistant

    # Time points
    t = np.linspace(0, duration, steps)

    # Solve the differential equations using logistic_growth_with_resistance
    solution = odeint(
        logistic_growth_with_resistance,
        [N_resistant, N_non_resistant],
        t,
        args=(r_resistant, r_non_resistant, K, loss_probability, mutation_rate),
    )

    return solution, t
