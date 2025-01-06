from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from tensorflow.keras.layers import Dense, Dropout
import joblib
from tensorflow.keras.optimizers import Adam

def build_mlp_model(input_shape, num_labels):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_labels, activation='sigmoid')  # Output a value for each label
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train Random Forest model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, '../models/model_random_forest.pkl')
    return model

# Evaluate the models and print detailed classification reports
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    # If MLP: convert probabilities to binary predictions
    if isinstance(model, Sequential):
        y_pred = (y_pred > 0.5).astype(int)

    # Calculate AUC for multi-label classification
    auc = roc_auc_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    # Print classification report for each label
    print(f"=== {model_name} Classification Report ===")
    for i, column in enumerate(y_test.columns):
        print(f"Label: {column}")
        print(classification_report(y_test[column], y_pred[:, i]))

    print(f"{model_name} AUC: {auc:.2f}")
    print(f"{model_name} Accuracy: {accuracy:.2f}\n")

    return auc, accuracy

# Main training and evaluation process
def test_model():
    from data_preprocessing import load_data, normalize_data

    # Load and split the data
    X, y = load_data()
    X = normalize_data(X)  # Normalize the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_auc, rf_accuracy = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Train MLP Neural Network
    mlp_model = build_mlp_model(input_shape=X_train.shape[1], num_labels=y_train.shape[1])
    mlp_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    mlp_auc, mlp_accuracy = evaluate_model(mlp_model, X_test, y_test, "MLP Neural Network")

    # Summary of model performance
    print("=== Model Performance Summary ===")
    print(f"Random Forest: Accuracy = {rf_accuracy:.2f}, AUC = {rf_auc:.2f}")
    print(f"MLP Neural Network: Accuracy = {mlp_accuracy:.2f}, AUC = {mlp_auc:.2f}")



if __name__ == '__main__':
    test_model()