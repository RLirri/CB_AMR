from data_preprocessing import load_data, plot_class_distribution
from ml_model2 import train_random_forest, build_mlp_model
from visualization import plot_roc_curves_for_all_antibiotics, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

"""cross-validation insights per antibiotic"""

# Load and preprocess data
X, y = load_data()
plot_class_distribution(y)

antibiotics = ['CIP', 'CTX', 'CTZ', 'GEN']
num_folds = 5  # Number of cross-validation folds
results = {antibiotic: {} for antibiotic in antibiotics}

# Initialize scaler and cross-validation
scaler = StandardScaler()
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Train models for each antibiotic
for antibiotic in antibiotics:
    print(f"Training and validating models for {antibiotic}...")
    y_target = y[antibiotic]

    results[antibiotic] = {'y_train': [], 'y_train_pred': [], 'y_val': [], 'y_val_pred': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_target)):
        print(f"  Fold {fold + 1}")

        # Split the data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_target.iloc[train_idx], y_target.iloc[val_idx]

        # Scale the data
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train Random Forest
        rf_model = train_random_forest(X_train, y_train)
        rf_train_pred = rf_model.predict_proba(X_train)[:, 1]
        rf_val_pred = rf_model.predict_proba(X_val)[:, 1]

        # Train MLP model
        mlp_model = build_mlp_model(X_train.shape[1])
        mlp_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
        mlp_train_pred = mlp_model.predict(X_train_scaled).ravel()
        mlp_val_pred = mlp_model.predict(X_val_scaled).ravel()

        # Track predictions for ROC
        results[antibiotic]['y_train'].extend(y_train)
        results[antibiotic]['y_train_pred'].extend(mlp_train_pred)
        results[antibiotic]['y_val'].extend(y_val)
        results[antibiotic]['y_val_pred'].extend(mlp_val_pred)



# Plot ROC curves for all antibiotics
plot_roc_curves_for_all_antibiotics(results)

# Cross-Validation with Stratified K-Folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_scores = cross_val_score(rf_model, X, y['CIP'], cv=cv, scoring='roc_auc')
print(f"Random Forest CV AUC Scores: {rf_scores}")
print(f"Mean AUC: {rf_scores.mean():.2f}")