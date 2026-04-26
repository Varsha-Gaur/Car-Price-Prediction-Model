# =============================================================================
# Car Price Prediction - End-to-End ANN Project
# Dataset: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset
# =============================================================================

# ---- Import Libraries -------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix,
                             classification_report)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import warnings
warnings.filterwarnings("ignore")

# Set a random seed so results are reproducible
np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
# STEP 1 — DATA LOADING
# =============================================================================
print("=" * 60)
print("STEP 1: Loading the Dataset")
print("=" * 60)

# Load the CSV file (place car data.csv in the same folder as this script)
df = pd.read_csv("car data.csv")

print(f"\nDataset Shape : {df.shape}")
print(f"Rows          : {df.shape[0]}")
print(f"Columns       : {df.shape[1]}")
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Basic Statistics ---")
print(df.describe())


# =============================================================================
# STEP 2 — DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Data Preprocessing")
print("=" * 60)

# --- 2a. Check for missing values ---
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with any missing values (there are very few, if any)
df.dropna(inplace=True)
print(f"\nShape after dropping missing values: {df.shape}")

# --- 2b. Drop unnecessary column ---
# Car_Name is just an identifier and won't help the model
df.drop(columns=["Car_Name"], inplace=True)
print("\nDropped 'Car_Name' column.")

# --- 2c. Encode categorical features ---
# Fuel_Type   : Petrol / Diesel / CNG
# Seller_Type : Dealer / Individual
# Transmission: Manual / Automatic

le = LabelEncoder()

df["Fuel_Type"]    = le.fit_transform(df["Fuel_Type"])
df["Seller_Type"]  = le.fit_transform(df["Seller_Type"])
df["Transmission"] = le.fit_transform(df["Transmission"])

print("\nAfter encoding categorical columns:")
print(df.head())

# --- 2d. Feature / Target split ---
X = df.drop(columns=["Selling_Price"])   # Features
y_reg = df["Selling_Price"]              # Regression target (continuous)

# Create classification target
# Low (0): < 3 lakh | Medium (1): 3–8 lakh | High (2): > 8 lakh
def price_category(price):
    if price < 3:
        return 0        # Low
    elif price <= 8:
        return 1        # Medium
    else:
        return 2        # High

y_clf = df["Selling_Price"].apply(price_category)
print("\nClassification label distribution:")
print(y_clf.value_counts().rename({0: "Low", 1: "Medium", 2: "High"}))

# --- 2e. Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures scaled using StandardScaler.")


# =============================================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Exploratory Data Analysis (EDA)")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EDA — Car Price Dataset", fontsize=16, fontweight="bold")

# Plot 1: Distribution of Selling Price
axes[0, 0].hist(df["Selling_Price"], bins=30, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Distribution of Selling Price")
axes[0, 0].set_xlabel("Selling Price (Lakhs)")
axes[0, 0].set_ylabel("Frequency")

# Plot 2: Selling Price vs Present Price
axes[0, 1].scatter(df["Present_Price"], df["Selling_Price"],
                   alpha=0.6, color="coral", edgecolors="none")
axes[0, 1].set_title("Present Price vs Selling Price")
axes[0, 1].set_xlabel("Present Price (Lakhs)")
axes[0, 1].set_ylabel("Selling Price (Lakhs)")

# Plot 3: Selling Price vs Kms Driven
axes[1, 0].scatter(df["Kms_Driven"], df["Selling_Price"],
                   alpha=0.6, color="mediumseagreen", edgecolors="none")
axes[1, 0].set_title("Kms Driven vs Selling Price")
axes[1, 0].set_xlabel("Kms Driven")
axes[1, 0].set_ylabel("Selling Price (Lakhs)")

# Plot 4: Selling Price by Fuel Type (0=CNG, 1=Diesel, 2=Petrol)
fuel_labels = {0: "CNG", 1: "Diesel", 2: "Petrol"}
for code, label in fuel_labels.items():
    subset = df[df["Fuel_Type"] == code]["Selling_Price"]
    axes[1, 1].hist(subset, bins=20, alpha=0.6, label=label)
axes[1, 1].set_title("Selling Price by Fuel Type")
axes[1, 1].set_xlabel("Selling Price (Lakhs)")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.show()
print("EDA plots saved as 'eda_plots.png'.")

# Correlation Heatmap
plt.figure(figsize=(9, 7))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
print("Correlation heatmap saved as 'correlation_heatmap.png'.")


# =============================================================================
# STEP 4 — TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Train / Test Split (80% train, 20% test)")
print("=" * 60)

# Regression split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42)

# Classification split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_clf, test_size=0.2, random_state=42)

# One-hot encode classification labels for softmax output
y_train_c_ohe = keras.utils.to_categorical(y_train_c, num_classes=3)
y_test_c_ohe  = keras.utils.to_categorical(y_test_c,  num_classes=3)

print(f"Regression  — Train: {X_train_r.shape}, Test: {X_test_r.shape}")
print(f"Classification — Train: {X_train_c.shape}, Test: {X_test_c.shape}")

num_features = X_scaled.shape[1]


# =============================================================================
# STEP 5 — BUILD ANN MODELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Building ANN Models")
print("=" * 60)

# --- Helper: build regression model ---
def build_regression_model(learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),        # Input layer
        layers.Dense(64, activation="relu"),         # Hidden layer 1
        layers.Dense(32, activation="relu"),         # Hidden layer 2
        layers.Dense(1, activation="linear")         # Output: continuous value
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model

# --- Helper: build classification model ---
def build_classification_model(learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),        # Input layer
        layers.Dense(64, activation="relu"),         # Hidden layer 1
        layers.Dense(32, activation="relu"),         # Hidden layer 2
        layers.Dense(3, activation="softmax")        # Output: 3 classes
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Print summaries
print("\n--- Regression Model Summary ---")
build_regression_model().summary()
print("\n--- Classification Model Summary ---")
build_classification_model().summary()


# =============================================================================
# STEP 6 — TRAIN MODELS (Default Hyperparameters)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Training Models (Default Hyperparameters)")
print("=" * 60)

# Default hyperparameters
DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR         = 0.001

print(f"\nEpochs: {DEFAULT_EPOCHS} | Batch Size: {DEFAULT_BATCH_SIZE} | LR: {DEFAULT_LR}")

# Train Regression Model
print("\n>>> Training Regression Model ...")
reg_model = build_regression_model(learning_rate=DEFAULT_LR)
reg_history = reg_model.fit(
    X_train_r, y_train_r,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    validation_split=0.1,
    verbose=0                  # Quiet mode — set to 1 to see epoch logs
)
print("Regression model training complete.")

# Train Classification Model
print("\n>>> Training Classification Model ...")
clf_model = build_classification_model(learning_rate=DEFAULT_LR)
clf_history = clf_model.fit(
    X_train_c, y_train_c_ohe,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    validation_split=0.1,
    verbose=0
)
print("Classification model training complete.")


# =============================================================================
# STEP 7 — EVALUATE MODELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Model Evaluation")
print("=" * 60)

# --- Regression Metrics ---
y_pred_r = reg_model.predict(X_test_r).flatten()

mse  = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
r2   = r2_score(y_test_r, y_pred_r)

print("\n--- Regression Results ---")
print(f"  MSE   : {mse:.4f}")
print(f"  RMSE  : {rmse:.4f}")
print(f"  R²    : {r2:.4f}")

# --- Classification Metrics ---
y_pred_prob = clf_model.predict(X_test_c)
y_pred_c    = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test_c, y_pred_c)

print("\n--- Classification Results ---")
print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n  Confusion Matrix:")
cm = confusion_matrix(y_test_c, y_pred_c)
print(cm)
print("\n  Classification Report:")
print(classification_report(y_test_c, y_pred_c,
                             target_names=["Low", "Medium", "High"]))

# --- Plot Training History ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Regression: loss curve
axes[0].plot(reg_history.history["loss"],     label="Train Loss")
axes[0].plot(reg_history.history["val_loss"], label="Val Loss", linestyle="--")
axes[0].set_title("Regression — Loss (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Classification: accuracy curve
axes[1].plot(clf_history.history["accuracy"],     label="Train Accuracy")
axes[1].plot(clf_history.history["val_accuracy"], label="Val Accuracy", linestyle="--")
axes[1].set_title("Classification — Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()
print("\nTraining history saved as 'training_history.png'.")

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved as 'confusion_matrix.png'.")


# =============================================================================
# STEP 8 — HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Hyperparameter Tuning Comparison")
print("=" * 60)

# We experiment by changing: Epochs, Batch Size, Learning Rate
# Each experiment trains a fresh regression model and measures R²

experiments = [
    {"epochs": 30,  "batch_size": 16,  "lr": 0.001,  "label": "Exp 1: LR=0.001, BS=16, E=30"},
    {"epochs": 50,  "batch_size": 32,  "lr": 0.001,  "label": "Exp 2: LR=0.001, BS=32, E=50  ← Default"},
    {"epochs": 100, "batch_size": 32,  "lr": 0.001,  "label": "Exp 3: LR=0.001, BS=32, E=100"},
    {"epochs": 50,  "batch_size": 64,  "lr": 0.001,  "label": "Exp 4: LR=0.001, BS=64, E=50"},
    {"epochs": 50,  "batch_size": 32,  "lr": 0.01,   "label": "Exp 5: LR=0.01,  BS=32, E=50"},
    {"epochs": 50,  "batch_size": 32,  "lr": 0.0001, "label": "Exp 6: LR=0.0001,BS=32, E=50"},
]

results = []

for exp in experiments:
    print(f"\n  Running: {exp['label']}")
    model = build_regression_model(learning_rate=exp["lr"])
    model.fit(
        X_train_r, y_train_r,
        epochs=exp["epochs"],
        batch_size=exp["batch_size"],
        validation_split=0.1,
        verbose=0
    )
    pred  = model.predict(X_test_r, verbose=0).flatten()
    r2_sc = r2_score(y_test_r, pred)
    rmse_sc = np.sqrt(mean_squared_error(y_test_r, pred))

    results.append({
        "label":      exp["label"],
        "epochs":     exp["epochs"],
        "batch_size": exp["batch_size"],
        "lr":         exp["lr"],
        "R2":         round(r2_sc, 4),
        "RMSE":       round(rmse_sc, 4),
    })
    print(f"    R² = {r2_sc:.4f} | RMSE = {rmse_sc:.4f}")

# Display comparison table
results_df = pd.DataFrame(results)
print("\n--- Hyperparameter Tuning Results ---")
print(results_df[["label", "R2", "RMSE"]].to_string(index=False))

# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels = [f"Exp {i+1}" for i in range(len(results))]

axes[0].bar(labels, results_df["R2"], color="steelblue", edgecolor="white")
axes[0].set_title("Hyperparameter Tuning — R² Score (higher = better)")
axes[0].set_ylabel("R² Score")
axes[0].set_ylim(0, 1)
for i, v in enumerate(results_df["R2"]):
    axes[0].text(i, v + 0.01, str(v), ha="center", fontsize=9)

axes[1].bar(labels, results_df["RMSE"], color="coral", edgecolor="white")
axes[1].set_title("Hyperparameter Tuning — RMSE (lower = better)")
axes[1].set_ylabel("RMSE (Lakhs)")
for i, v in enumerate(results_df["RMSE"]):
    axes[1].text(i, v + 0.01, str(v), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("hyperparameter_comparison.png", dpi=150)
plt.show()
print("\nHyperparameter comparison chart saved as 'hyperparameter_comparison.png'.")

print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)
print("\nGenerated files:")
print("  eda_plots.png")
print("  correlation_heatmap.png")
print("  training_history.png")
print("  confusion_matrix.png")
print("  hyperparameter_comparison.png")
