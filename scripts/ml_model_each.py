from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Train Random Forest model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, '../models/model_random_forest.pkl')
    return model

# Build MLP Neural Network model for binary classification
def build_mlp_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Output 1 neuron for binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Evaluate model and print performance metrics
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    # If MLP: Convert probabilities to binary predictions
    if isinstance(model, Sequential):
        y_pred = (y_pred > 0.5).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Print detailed classification report
    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} AUC: {auc:.2f}")
    print(f"{model_name} Accuracy: {accuracy:.2f}\n")

    return auc, accuracy

# Plot Confusion Matrix for visualization
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Train and evaluate models per antibiotic
def train_and_evaluate_per_antibiotic(X, y):
    antibiotics = y.columns  # ['CIP', 'CTX', 'CTZ', 'GEN']
    results = {}

    for antibiotic in antibiotics:
        print(f"\nTraining models for {antibiotic}...")

        y_target = y[antibiotic]

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)

        # Train Random Forest
        rf_model = train_random_forest(X_train, y_train)
        rf_auc, rf_accuracy = evaluate_model(rf_model, X_test, y_test, f"Random Forest - {antibiotic}")
        plot_confusion_matrix(y_test, rf_model.predict(X_test), f"Random Forest - {antibiotic}")

        # Train MLP Model
        mlp_model = build_mlp_model(X_train.shape[1])
        mlp_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        mlp_auc, mlp_accuracy = evaluate_model(mlp_model, X_test, y_test, f"MLP Neural Network - {antibiotic}")
        plot_confusion_matrix(y_test, (mlp_model.predict(X_test) > 0.5).astype(int), f"MLP Neural Network - {antibiotic}")

        # Store results
        results[antibiotic] = {
            'Random Forest': {'AUC': rf_auc, 'Accuracy': rf_accuracy},
            'MLP Neural Network': {'AUC': mlp_auc, 'Accuracy': mlp_accuracy}
        }

    return results

# Summary plot of model performance for all antibiotics
def plot_overall_performance(results):
    antibiotics = results.keys()
    rf_accuracies = [results[ab]['Random Forest']['Accuracy'] for ab in antibiotics]
    mlp_accuracies = [results[ab]['MLP Neural Network']['Accuracy'] for ab in antibiotics]

    x = np.arange(len(antibiotics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, rf_accuracies, width, label='Random Forest')
    ax.bar(x + width/2, mlp_accuracies, width, label='MLP Neural Network')

    ax.set_xlabel('Antibiotics')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison by Antibiotic')
    ax.set_xticks(x)
    ax.set_xticklabels(antibiotics)
    ax.legend()

    plt.show()

# Main function to load data, train, and evaluate models
def main():
    from data_preprocessing import load_data, normalize_data

    # Load and preprocess data
    X, y = load_data()
    X = normalize_data(X)

    # Train and evaluate models for each antibiotic
    results = train_and_evaluate_per_antibiotic(X, y)

    # Plot overall performance
    plot_overall_performance(results)

    print("\n=== Final Model Performance Summary ===")
    for antibiotic, metrics in results.items():
        print(f"\nAntibiotic: {antibiotic}")
        for model_name, performance in metrics.items():
            print(f"{model_name} - Accuracy: {performance['Accuracy']:.2f}, AUC: {performance['AUC']:.2f}")

if __name__ == '__main__':
    main()
