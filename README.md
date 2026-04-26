# 🚗 Car Price Prediction 

A Machine Learning project that uses an **Artificial Neural Network (ANN)** to predict car prices. This project covers the full ML pipeline: data loading, preprocessing, EDA, model building, evaluation, and hyperparameter tuning.

---

## 📂 Project Structure

```
car_price_ann/
│
├── main.py                      # Main Python script (all steps)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── car data.csv                 # Dataset (download from Kaggle — see below)
│
└── (generated after running)
    ├── eda_plots.png
    ├── correlation_heatmap.png
    ├── training_history.png
    ├── confusion_matrix.png
    └── hyperparameter_comparison.png
```

---

## 📊 Dataset

**Source:** [Car Price Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset)

| Column          | Description                                    |
|-----------------|------------------------------------------------|
| `Car_Name`      | Name of the car (dropped during preprocessing) |
| `Year`          | Year the car was purchased                     |
| `Selling_Price` | Price at which the car is being sold (target)  |
| `Present_Price` | Current showroom price of the car              |
| `Kms_Driven`    | Total kilometres driven                        |
| `Fuel_Type`     | Petrol / Diesel / CNG                          |
| `Seller_Type`   | Dealer / Individual                            |
| `Transmission`  | Manual / Automatic                             |
| `Owner`         | Number of previous owners                      |

**Size:** ~301 rows × 9 columns

---

## 🧠 What This Project Covers

### 1. Data Loading
- Load CSV with pandas
- View shape, info, and statistics

### 2. Data Preprocessing
- Drop `Car_Name` (not useful for prediction)
- Label encode categorical columns (`Fuel_Type`, `Seller_Type`, `Transmission`)
- Handle missing values
- Scale features using `StandardScaler`

### 3. Exploratory Data Analysis (EDA)
- Distribution of Selling Price
- Scatter plots (feature vs target)
- Correlation heatmap

### 4. Two ML Tasks
| Task           | Target                                   |
|----------------|------------------------------------------|
| Regression     | Predict exact Selling Price              |
| Classification | Predict Low / Medium / High price bucket |

Price buckets:
- **Low (0):** < 3 Lakh
- **Medium (1):** 3–8 Lakh
- **High (2):** > 8 Lakh

### 5. ANN Architecture (TensorFlow/Keras)

Both models share the same backbone:

```
Input → Dense(64, ReLU) → Dense(32, ReLU) → Output
```

| Model          | Output Activation | Loss Function          |
|----------------|-------------------|------------------------|
| Regression     | Linear            | Mean Squared Error     |
| Classification | Softmax (3 units) | Categorical Crossentropy |

### 6. Evaluation Metrics
- **Regression:** MSE, RMSE, R² Score
- **Classification:** Accuracy, Confusion Matrix, Classification Report

### 7. Hyperparameter Tuning
Six experiments varying Epochs, Batch Size, and Learning Rate:

| Experiment | LR     | Batch Size | Epochs |
|------------|--------|------------|--------|
| Exp 1      | 0.001  | 16         | 30     |
| Exp 2 ✅   | 0.001  | 32         | 50     |
| Exp 3      | 0.001  | 32         | 100    |
| Exp 4      | 0.001  | 64         | 50     |
| Exp 5      | 0.01   | 32         | 50     |
| Exp 6      | 0.0001 | 32         | 50     |

Results are compared using R² Score and RMSE.

---

## ⚙️ How to Download the Dataset

1. Go to: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset
2. Download `car data.csv`
3. Place it in the **same folder** as `main.py`


## 📈 Results Summary

> Note: Results may vary slightly depending on random seed and TensorFlow version.

### Regression (Predicting Selling Price)
| Metric | Score  |
|--------|--------|
| MSE    | ~1.8   |
| RMSE   | ~1.35  |
| R²     | ~0.93  |

### Classification (Low / Medium / High)
| Metric   | Score   |
|----------|---------|
| Accuracy | ~90–95% |

### Hyperparameter Tuning
- Best R² is typically achieved around **Exp 3** (more epochs = better convergence)
- Very high LR (0.01) can make the model unstable
- Very low LR (0.0001) under-trains the model in 50 epochs

---


## 🛠️ Tech Stack

| Tool          | Version  | Purpose               |
|---------------|----------|-----------------------|
| Python        | 3.10+    | Programming language  |
| pandas        | 2.2.2    | Data loading & wrangling |
| numpy         | 1.26.4   | Numerical operations  |
| matplotlib    | 3.8.4    | Plotting              |
| seaborn       | 0.13.2   | Statistical plots     |
| scikit-learn  | 1.4.2    | Preprocessing & metrics |
| TensorFlow    | 2.16.1   | Building & training ANN |

---


## 📝 License

This project is for educational purposes. Dataset is from Kaggle — refer to their terms of use.
