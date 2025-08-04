# 🏥 PredictLOS – Predicting Patient Length of Stay

This project focuses on using machine learning models to predict the **Length of Stay (LOS)** of patients in hospitals using structured patient and hospital-level data.

---
## 📌 Objective

The primary goal of this project is to **predict the Length of Stay** (LOS) for patients using hospital and patient-related features. Predicting LOS can help hospitals:

- Optimize resources,
- Improve treatment efficiency,
- Reduce infection rates, and
- Enhance patient care outcomes.

---
## 📊 Dataset Overview

- 📦 **Total records:** 318,438
- 🏷️ **Target Variable:** `Stay` (Categorical with 11 classes, e.g., "0-10", "11-20", ..., ">100 Days")
- 👥 **Features:** 17 columns including patient ID, hospital code, admission type, age, illness severity, etc.

| Feature                    | Description                     |
| -------------------------- | ------------------------------- |
| `Hospital_code`            | Unique identifier for hospitals |
| `Type of Admission`        | Urgent / Emergency / Trauma     |
| `Severity of Illness`      | Minor / Moderate / Extreme      |
| `Visitors with Patient`    | Numerical                       |
| `Age`                      | Categorical ranges              |
| `Admission_Deposit`        | Amount deposited at admission   |
| `Ward Type` & `Department` | Hospital resource indicators    |
| ...                        | and more                        |

---

## 🧹 Data Preparation & Feature Engineering

- **Missing Values Imputed** using mode for `Bed Grade` and `City_Code_Patient`
- **Label Encoding** applied to categorical variables
- Created new engineered features:
  - `count_id_patient`
  - `count_id_patient_hospitalCode`
  - `count_id_patient_wardfacilityCode`

---

## 🧠 Models Applied

### 1. **Naïve Bayes (Gaussian)**

- Probabilistic model using Bayes’ Theorem
- **Accuracy:** `34.55%`
- **Observation:** Biased toward predicting the 21–30 day class

### 2. **XGBoost Classifier**

- Tuned hyperparameters:
  - `max_depth=4`, `n_estimators=800`, `learning_rate=0.1`
  - `reg_alpha=0.5`, `reg_lambda=1.5`, `min_child_weight=2`
- **Accuracy:** `43.05%`
- **Observation:** Best performance among all models

### 3. **Neural Network (Keras Sequential)**

- 6 dense layers with ReLU activations and final softmax layer
- Trained with:
  - `categorical_crossentropy` loss
  - `SGD` optimizer
- **Accuracy:** `42.05%`
- **Observation:** Slightly lower than XGBoost, limited by dataset size

---

## 📈 Model Performance Summary

| Length of Stay | Naïve Bayes | XGBoost | Neural Net |
| -------------- | ----------- | ------- | ---------- |
| 0–10 Days      | 2,598       | 4,373   | 4,517      |
| 11–20 Days     | 26,827      | 39,337  | 35,982     |
| 21–30 Days     | 72,206      | 58,261  | 61,911     |
| 31–40 Days     | 15,639      | 12,100  | 8,678      |
| 41–50 Days     | 469         | 61      | 26         |
| 51–60 Days     | 13,651      | 19,217  | 21,709     |
| 61–70 Days     | 92          | 16      | 1          |
| 71–80 Days     | 955         | 302     | 248        |
| 81–90 Days     | 296         | 1,099   | 1,165      |
| 91–100 Days    | 2           | 78      | 21         |
| >100 Days      | 4,322       | 2,213   | 2,799      |

---

## 🔮 Future Insights

- **Smart Staffing:** Better workforce planning through demand forecasting
- **Disease Risk Management:** Early intervention and personalized care
- **Real-time Clinical Alerts:** On-the-spot recommendations via CDS tools
- **Patient Engagement:** Leverage wearable data to track vitals and habits

---

## 🧾 Repository Contents

- `Healthcare_Analytics.ipynb`: Code and model pipeline
- `HTML_Report.html`: Rendered notebook with output
- `datasets.zip`: Training and testing datasets
- `TensorBoard.png`: Training logs for the Neural Network

---

## 📚 References

- Analytics Vidhya – JanataHack Dataset
- XGBoost Documentation
- Blogs and articles on Naïve Bayes, Neural Networks, and Healthcare Analytics
- TensorBoard visualizations for NN training
