from data_preprocessing import load_data, plot_class_distribution
from ml_models import train_random_forest, build_mlp_model
from mechanistic_model import simulate_logistic_growth
from abm import simulate_abm
# from advanced_bio_analysis import gene_level_resistance_analysis
from hyperparameter_tuning import grid_search_rf
from visualization import plot_confusion_matrix, plot_roc_curve, plot_model_comparison, plot_accuracy_graphs
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Load and preprocess data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
plot_class_distribution(y)

# Train Random Forest model
rf_model = train_random_forest(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Hyperparameter Tuning: Random Forest
rf_model = grid_search_rf(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Train MLP model: with Scaled Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = build_mlp_model(X_train.shape[1])
mlp_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1)
mlp_pred = (mlp_model.predict(X_test_scaled) > 0.5).astype(int)

# Visualizations
plot_confusion_matrix(y_test.values, rf_pred, "Random Forest")
plot_confusion_matrix(y_test.values, mlp_pred, "MLP Neural Network")
rf_auc = plot_roc_curve(y_test, rf_pred, "Random Forest")
mlp_auc = plot_roc_curve(y_test, mlp_pred, "MLP Neural Network")

# Model Comparison Plot
models = ['Random Forest', 'MLP Neural Network']
accuracies = [accuracy_score(y_test, rf_pred), accuracy_score(y_test, mlp_pred)]
aucs = [rf_auc, mlp_auc]
plot_model_comparison(models, accuracies, aucs)

# Feature importance for Random Forest
feature_importances = rf_model.feature_importances_
important_features = X.columns[feature_importances > 0.1]  # Threshold for significance

# Output Results Summary
print("=== Model Performance Summary ===")
for i, model in enumerate(models):
    print(f"{model}: Accuracy = {accuracies[i]:.2f}, AUC = {aucs[i]:.2f}")

better_model = models[accuracies.index(max(accuracies))]
print(f"\n{better_model} performs better overall based on accuracy.")

#  Print Biological Insights
def generate_biological_insights(accuracies, aucs, important_features):
    print("""
    The results provide insights into antibiotic resistance across four antibiotics: CIP, CTX, CTZ, and GEN.
    1. MLP Neural Network is useful for capturing non-linear patterns, which may reflect complex gene interactions.
    2. Random Forest is more interpretable, showing which features (e.g., specific resistance genes) are important.
    3. The high AUC score in either model suggests that our model can effectively distinguish between resistant and non-resistant strains.
    4. Resistance patterns across antibiotics might indicate cross-resistance, a phenomenon where resistance to one antibiotic leads to resistance to others. This aligns with known biological behaviors in resistant bacterial strains.
    """)
    insights = ""

    # Compare models based on AUC
    if aucs[0] > aucs[1]:
        insights += f"Random Forest has a better AUC ({aucs[0]:.2f}), indicating better handling of class imbalance.\n"
    else:
        insights += f"MLP Neural Network achieves a higher AUC ({aucs[1]:.2f}), capturing non-linear patterns.\n"

    # Check feature importance in Random Forest
    if len(important_features) > 0:
        insights += "Important features influencing the model include:\n"
        for feature in important_features:
            insights += f"- {feature}: potential biological marker for antibiotic resistance.\n"
    else:
        insights += "No dominant features were identified, suggesting complex interactions among variables.\n"

    # Analyze cross-resistance
    if accuracies[0] > 0.85 and accuracies[1] > 0.85:
        insights += "High accuracies across models suggest cross-resistance trends.\n"

    # Mechanistic model insight suggestion
    if better_model == "Random Forest":
        insights += "Random Forest’s interpretability suggests further mechanistic modeling to understand feature interactions.\n"
    else:
        insights += "MLP's ability to capture non-linearity suggests extending agent-based modeling for deeper exploration.\n"

    return insights

insights = generate_biological_insights(accuracies, aucs, important_features)
print("\n=== Biological Insights ===")
print(insights)

# Mechanistic Model and ABM Simulation
simulate_logistic_growth(y_train)
# logistic growth simulation will now extract resistance ratios from the training
# dataset (y_train), which adjusts the growth rate (r) and carrying capacity (K) dynamically.
simulate_abm()
