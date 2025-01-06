import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
"""
    Can ignore this file first
    """
def gene_level_resistance_analysis(X, y):
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])

    # Handle y based on its type and shape
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.iloc[:, 0]  # Extract the first column as a Series
        else:
            raise ValueError("y DataFrame must have only one column.")
    elif isinstance(y, np.ndarray):
        if y.ndim == 2 and y.shape[1] == 1:
            y = pd.Series(y[:, 0], name="Resistance")  # Extract column as Series
        elif y.ndim == 1:
            y = pd.Series(y, name="Resistance")
        else:
            raise ValueError("y ndarray must be 1D or 2D with one column.")
    elif not isinstance(y, pd.Series):
        y = pd.Series(y, name="Resistance")

    # Plot correlation matrix using Seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix of Features")
    plt.show()

    # Merge X and y into a single DataFrame for Plotly scatter matrix
    df = pd.concat([X, y], axis=1)

    # Plot scatter matrix with Plotly
    fig = px.scatter_matrix(
        df,
        dimensions=X.columns,
        color="Resistance",
        title="Gene-Level Resistance Prediction",
        labels={col: col for col in X.columns},
        height=800, width=800
    )
    fig.update_traces(diagonal_visible=False)  # Optional: Hide diagonal histograms
    fig.show()
