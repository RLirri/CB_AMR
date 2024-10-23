import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

"""
    To Test the test.py, but now have the TensorFlow model's input and output shapes not aligning error
    Can ignore this file first
    """

# === 1. Load and preprocess the dataset === #
def load_data(file_path='../dataset/gi_cip_ctx_ctz_gen_pheno.csv', test_size=0.2, random_state=42):
    """
    Load the dataset, handle missing values if any, and return train-test splits.

    Args:
        file_path: Path to the dataset file.
        test_size: Ratio of the test dataset size.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets for training and testing.
    """
    data = pd.read_csv(file_path)

    # Handle missing values if any
    data.fillna(0, inplace=True)  # Replace missing values with 0, adjust if needed

    # Separate features (X) and labels (y)
    X = data.drop(columns=['prename'])
    y = data[['CIP', 'CTX', 'CTZ', 'GEN']]

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# === 2. Feature Normalization === #
def normalize_data(X_train, X_test):
    """
    Standardize features by removing the mean and scaling to unit variance.

    Args:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.

    Returns:
        X_train_scaled, X_test_scaled: Normalized feature matrices.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), \
        pd.DataFrame(X_test_scaled, columns=X_test.columns)


# === 3. Plot Class Distribution === #
def plot_class_distribution(y):
    """
    Plot the distribution of antibiotic resistance classes.

    Args:
        y: Label matrix with binary resistance indicators for CIP, CTX, CTZ, and GEN.
    """
    plt.figure(figsize=(10, 6))
    y.sum(axis=0).plot(kind='bar', color='skyblue')
    plt.title("Class Distribution of Antibiotic Resistance", fontsize=16)
    plt.ylabel("Number of Resistant Samples")
    plt.xlabel("Antibiotics")
    plt.xticks(rotation=45)
    plt.show()


# === 4. Correlation Matrix of Features === #
def plot_correlation_matrix(X):
    """
    Plot the correlation matrix to analyze relationships between features.

    Args:
        X: Feature matrix.
    """
    corr_matrix = X.corr()
    plt.figure(figsize=(10, 8))
    plt.title("Feature Correlation Matrix", fontsize=16)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.show()


# === 5. Explore Missing Data === #
def plot_missing_data(data):
    """
    Plot the percentage of missing values in each feature.

    Args:
        data: Original dataset before splitting.
    """
    missing_data = data.isnull().mean() * 100
    plt.figure(figsize=(12, 6))
    missing_data.plot(kind='bar', color='tomato')
    plt.title("Missing Data Percentage by Feature", fontsize=16)
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.show()


# === 6. Antibiotic Resistance Heatmap === #
def plot_resistance_heatmap(y):
    """
    Plot a heatmap to visualize resistance patterns across samples.

    Args:
        y: Label matrix (binary resistance indicators).
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(y, cmap='YlGnBu', cbar=True, xticklabels=y.columns, yticklabels=False)
    plt.title("Resistance Pattern Heatmap", fontsize=16)
    plt.xlabel("Antibiotics")
    plt.show()
