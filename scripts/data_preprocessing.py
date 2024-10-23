import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Load and preprocess the dataset
def load_data(file_path='../dataset/gi_cip_ctx_ctz_gen_pheno.csv'):
    """
    Load the dataset and return feature matrix X and label matrix y.

    CIP: Ciprofloxacin
    CTX: Cefotaxime
    CTZ: Ceftazidime
    GEN: Gentamicin
    """
    data = pd.read_csv(file_path)
    X = data.drop(columns=['prename'])
    y = data[['CIP', 'CTX', 'CTZ', 'GEN']]
    return X, y


def normalize_data(X):
    """
    Standardize features by removing the mean and scaling to unit variance.
    This ensures that all features contribute equally to the model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)


def plot_class_distribution(y):
    """
    Plot the distribution of antibiotic resistance classes.
    """
    plt.figure(figsize=(10, 6))
    y.sum(axis=0).plot(kind='bar', color='skyblue')
    plt.title("Class Distribution of Antibiotic Resistance")
    plt.ylabel("Number of Resistant Samples")
    plt.xlabel("Antibiotics")
    plt.xticks(rotation=45)
    plt.show()


def plot_correlation_matrix(X):
    """
    Plot the correlation matrix of features to identify relationships.
    """
    corr_matrix = X.corr()
    plt.figure(figsize=(10, 8))
    plt.title("Feature Correlation Matrix", fontsize=16)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.show()
