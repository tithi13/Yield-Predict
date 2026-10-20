# Crop Yield Prediction using Weather and Soil Data

A machine learning prototype developed strictly following academic guidelines for a 5th-semester Information Technology capstone/viva project.

---

## 📌 Project Overview
Predicting agricultural crop yield based on environmental factors (rainfall, temperature, humidity) and soil nutrient indicators (soil pH, Nitrogen, Phosphorus, Potassium). The project compares **Linear Regression** and **Random Forest Regressor** on the exact same held-out test dataset, strictly preventing data leakage by fitting all scalers and one-hot encoders exclusively on training partitions.

---

## 🛠️ Technology Stack
- **Programming Language:** Python 3.13
- **Data Manipulation & Math:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
- **Model Serialization:** Joblib
- **Web Application & UI:** Streamlit
- **Data Visualizations:** Plotly Express & Plotly Graph Objects

---

## 📂 Project Structure
```text
e:\Yield-Predict\
│
├── data/
│   ├── crop_yield.csv                      # Raw historical crop records
│   ├── state_soil_data.csv                 # Raw soil nutrient profiles (N, P, K, pH)
│   ├── state_weather_data_1997_2020.csv    # Raw meteorological records
│   ├── processed_crop_yield_data.csv       # Cleaned merged historical dataset (19,517 rows)
│   └── README.md                           # Dataset provenance & acquisition details
│
├── models/
│   ├── linear_regression_pipeline.joblib   # Trained Linear Regression pipeline
│   ├── random_forest_pipeline.joblib       # Trained Random Forest pipeline
│   ├── model_metadata.json                 # Model metrics, feature names, & units
│   └── test_predictions_sample.csv        # Held-out actual vs predicted records
│
├── app.py                                  # Multi-page interactive Streamlit dashboard
├── train_model.py                          # Data cleaning, split, training & eval pipeline
├── test_pipeline.py                       # Automated verification & inference test suite
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation & viva guide
```

---

## 📊 Dataset Provenance & Target Variable
- **Source:** Kaggle (`anshumish/crop-yield-data-with-soil-and-weather-dataset`), compiled from Ministry of Agriculture & Farmers Welfare (Govt. of India) and regional meteorological data.
- **Records:** 19,517 historical observations covering 54 crop varieties from 1997 to 2020.
- **Target Variable:** `yield` (Historical Crop Yield).
- **Yield Units:** **Tonnes per Hectare (t/ha)**.
- **Input Features (8 Required):**
  1. `rainfall` — Total Annual Rainfall in millimeters (mm)
  2. `temperature` — Mean Temperature in degrees Celsius (°C)
  3. `humidity` — Mean Relative Humidity in percentage (%)
  4. `pH` — Soil pH level (0–14)
  5. `N` — Soil Nitrogen content (kg/ha)
  6. `P` — Soil Phosphorus content (kg/ha)
  7. `K` — Soil Potassium content (kg/ha)
  8. `crop_type` — One-Hot Encoded Categorical Crop (54 classes, e.g., Rice, Wheat, Maize, Cotton)

*Unit Consistency Note:* In Indian agricultural datasets, coconut production is measured in *thousands of nuts per hectare* rather than metric tonnes. To maintain rigorous unit consistency (`Tonnes per Hectare`) across all algorithms and dashboard visualizations, Coconut records were explicitly isolated.

---

## 📈 Model Performance & Evaluation Results
Evaluated on a strictly isolated 20% held-out test set ($n = 3,904$ samples) using 4 standard regression metrics:

| Metric | Linear Regression | Random Forest Regressor | Preferred / Better Model |
| :--- | :---: | :---: | :---: |
| **MAE (Mean Absolute Error)** | `2.8152 t/ha` | **`1.2291 t/ha`** | **Random Forest (56.3% lower error)** |
| **MSE (Mean Squared Error)** | `41.5914` | **`24.3183`** | **Random Forest** |
| **RMSE (Root Mean Squared Error)** | `6.4491 t/ha` | **`4.9314 t/ha`** | **Random Forest (23.5% lower error)** |
| **R² Score (Coefficient of Determination)** | `0.7367` | **`0.8460`** | **Random Forest (Explains 84.6% of variance)** |

### Why Random Forest Outperformed Linear Regression:
1. **Non-Linear Agricultural Dynamics:** Biological crop response to environmental conditions follows non-linear bell curves (extreme rainfall causes waterlogging; extreme temperatures cause heat stress). Linear regression enforces a rigid hyper-plane.
2. **Feature Interdependence:** Soil pH directly impacts the bioavailability of Nitrogen, Phosphorus, and Potassium. Decision trees partition these high-order interactions more effectively than additive linear equations.

---

## 🚀 Installation & Exact Execution Commands (Windows)

### 1. Open Terminal / PowerShell
```powershell
cd e:\Yield-Predict
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Automated Tests
```powershell
python test_pipeline.py
```

### 4. Train Models & Generate Artifacts (Optional, already pre-trained)
```powershell
python train_model.py
```

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
*Access the dashboard in your web browser at: `http://localhost:8501`*

---

## 🌿 Heuristic Fertilizer Advisory Feature
The application includes an educational nutrient balance check based on agronomic heuristics for N, P, K, and soil pH.
> **⚠️ Mandatory Disclaimer:** This feature is provided purely as an educational heuristic advisory and **NOT** as a scientifically validated fertilizer prescription. It must not be used for real-world farming interventions without physical laboratory soil testing and guidance from certified agricultural extension authorities.

---

## ⚠️ Known Limitations & Scope
1. **Regional Level Aggregations:** Meteorological and soil indicators are compiled at state/district annual levels. Field-level microclimates and precision soil samples may exhibit localized variances.
2. **Unmodeled Exogenous Variables:** Unrecorded factors like seed cultivar genetics (hybrid vs traditional), pest outbreaks, micro-irrigation schedules, and sudden unseasonal hail storms are outside the dataset's scope.
3. **Real-World Farm Usage:** The application is an academic prototype for viva demonstration and not an agronomic decision-support platform.
