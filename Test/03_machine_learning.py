# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# Load the dataset
data = pd.read_csv('data/gi_cip_ctx_ctz_gen_pheno.csv')

# Step 1: Data Preprocessing
# Extract features and labels
X = data.drop(columns=['prename'])  # Features
y = data[['CIP', 'CTX', 'CTZ', 'GEN']]  # Labels (Multi-label targets)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Exploratory Data Analysis (EDA)
print("Dataset Overview:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe())

# Check class distribution for each label
print("\nClass Distribution:")
print(y.sum(axis=0))  # Number of resistant samples per antibiotic

# Step 3: Random Forest Model
print("\nTraining Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest model
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['CIP', 'CTX', 'CTZ', 'GEN']))

# Save the Random Forest model
joblib.dump(rf_model, 'models/model_random_forest.pkl')

# Step 4: Neural Network (MLP) Model
print("\nTraining MLP Neural Network...")
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the MLP model structure
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='sigmoid')  # Output layer with 4 nodes for multi-label classification
])

# Compile the model
mlp_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the model
mlp_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Predict on the test set
mlp_pred = (mlp_model.predict(X_test_scaled) > 0.5).astype(int)

# Evaluate MLP model
print("\nMLP Neural Network Classification Report:")
print(classification_report(y_test, mlp_pred, target_names=['CIP', 'CTX', 'CTZ', 'GEN']))

# Save the MLP model
mlp_model.save('models/model_mlp.h5')

# Step 5: Summary and Comparison of Models
rf_accuracy = accuracy_score(y_test, rf_pred)
mlp_accuracy = accuracy_score(y_test, mlp_pred)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.2f}")
print(f"MLP Accuracy: {mlp_accuracy:.2f}")

# Conclusion on model selection
if rf_accuracy > mlp_accuracy:
    print("Random Forest performed better.")
else:
    print("MLP Neural Network performed better.")
