from data_processing_normalization import (
    load_data, normalize_data, plot_class_distribution,
    plot_correlation_matrix, plot_missing_data, plot_resistance_heatmap
)
from ml_model_each import train_random_forest, build_mlp_model
from hyperparameter_tuning import grid_search_rf
from visualization import (
    plot_confusion_matrix_each, plot_roc_curve,
    plot_accuracy_graphs_each
)
from abm import simulate_abm
from mechanistic_model import simulate_logistic_growth
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

"""Don't run this file, has error"""

# Load dataset and split into train/test
X_train, X_test, y_train, y_test = load_data()

# Normalize features
X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

# Visualizations for Data Diagnostics
plot_class_distribution(y_train)
plot_missing_data(X_train)
plot_correlation_matrix(X_train)
plot_resistance_heatmap(y_train)

# List of antibiotics to analyze independently
antibiotics = ['CIP', 'CTX', 'CTZ', 'GEN']
num_folds = 5  # Number of cross-validation folds

# Dictionary to store model results per antibiotic
results = {antibiotic: {'train_accuracy': [], 'val_accuracy': [],
                        'train_auc': [], 'val_auc': []} for antibiotic in antibiotics}

# Cross-validation setup
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
scaler = StandardScaler()

# === Train models and visualize results per antibiotic === #
for antibiotic in antibiotics:
    print(f"\nTraining models for {antibiotic}...")

    # Target labels for the current antibiotic
    y_target_train = y_train[antibiotic]
    y_target_test = y_test[antibiotic]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_target_train)):
        print(f"  Fold {fold + 1}")

        # Split the data into training and validation sets
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_target_train.iloc[train_idx], y_target_train.iloc[val_idx]

        # Scale the features
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = scaler.transform(X_fold_val)

        # Train Random Forest
        rf_model = train_random_forest(X_fold_train, y_fold_train)
        rf_train_pred = rf_model.predict(X_fold_train)
        rf_val_pred = rf_model.predict(X_fold_val)

        # Train MLP Neural Network
        mlp_model = build_mlp_model(X_train.shape[1])
        mlp_model.fit(X_fold_train_scaled, y_fold_train, epochs=10, batch_size=32, verbose=0)
        mlp_train_pred = (mlp_model.predict(X_fold_train_scaled) > 0.5).astype(int)
        mlp_val_pred = (mlp_model.predict(X_fold_val_scaled) > 0.5).astype(int)

        # Calculate Accuracy and AUC
        rf_train_accuracy = accuracy_score(y_fold_train, rf_train_pred)
        rf_val_accuracy = accuracy_score(y_fold_val, rf_val_pred)
        rf_auc = roc_auc_score(y_fold_val, rf_model.predict_proba(X_fold_val)[:, 1])

        mlp_train_accuracy = accuracy_score(y_fold_train, mlp_train_pred)
        mlp_val_accuracy = accuracy_score(y_fold_val, mlp_val_pred)
        mlp_auc = roc_auc_score(y_fold_val, mlp_model.predict(X_fold_val_scaled).ravel())

        # Store results
        results[antibiotic]['train_accuracy'].append(rf_train_accuracy)
        results[antibiotic]['val_accuracy'].append(rf_val_accuracy)
        results[antibiotic]['train_auc'].append(rf_auc)
        results[antibiotic]['val_auc'].append(mlp_auc)

    # Visualize Confusion Matrices and ROC Curves
    print(f"\nVisualizing performance for {antibiotic}...")
    plot_confusion_matrix_each(y_target_test.values, rf_model.predict(X_test), f"{antibiotic} - Random Forest")
    plot_confusion_matrix_each(y_target_test.values, (mlp_model.predict(X_test_scaled) > 0.5).astype(int),
                               f"{antibiotic} - MLP Neural Network")
    plot_roc_curve(y_target_test, rf_model.predict_proba(X_test)[:, 1], f"{antibiotic} - Random Forest")
    plot_roc_curve(y_target_test, mlp_model.predict(X_test_scaled).ravel(), f"{antibiotic} - MLP Neural Network")

# Visualize overall performance comparison
print("\nPlotting overall model performance comparison...")
plot_accuracy_graphs_each(results, num_folds)

# Hyperparameter Tuning: Random Forest on one antibiotic
best_rf_model = grid_search_rf(X_train, y_train['CIP'])

# Summary of model performance
print("\n=== Model Performance Summary ===")
for antibiotic in antibiotics:
    mean_train_accuracy = np.mean(results[antibiotic]['train_accuracy'])
    mean_val_accuracy = np.mean(results[antibiotic]['val_accuracy'])
    print(f"{antibiotic} - Train Accuracy: {mean_train_accuracy:.2f}, Validation Accuracy: {mean_val_accuracy:.2f}")

# Mechanistic and Agent-Based Modeling Simulations
print("\nRunning Mechanistic and Agent-Based Models...")
simulate_logistic_growth(y_train)
simulate_abm()
