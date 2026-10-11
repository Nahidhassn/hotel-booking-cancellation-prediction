# Hotel Booking Cancellation Prediction 🏨

A machine learning project that predicts whether a hotel booking will be **canceled**, comparing **Logistic Regression, Random Forest and Gradient Boosting**, with careful handling of **data leakage**, duplicate records, missing values, feature engineering, threshold analysis and permutation-based feature importance.

---

## 🎯 Problem Statement

Cancellations cost hotels revenue: rooms sit empty that could have been resold. If a hotel can estimate **which bookings are likely to be canceled**, it can manage overbooking more safely, target risky bookings with reminders or deposit policies, and plan staffing and inventory better.

**Dataset:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) — 119,390 bookings from a Resort Hotel and a City Hotel (2015–2017), 32 columns.

---

## 🔍 Project Workflow

| Stage | What was done |
|---|---|
| **Data Cleaning** | Removed 31,994 duplicate rows, dropped invalid rows (166 bookings with zero guests, 1 with negative ADR), filled missing `children`, `country` and `agent` values |
| **Leakage Prevention** | Removed `reservation_status` and `reservation_status_date` (they directly reveal the outcome) and `assigned_room_type` (not known when the booking is made). Dropped `company` (~94% missing) |
| **Feature Engineering** | Created `total_nights` and `total_guests` |
| **EDA** | Cancellation rate by hotel type, deposit type, lead time, market segment, arrival month, special requests and previous cancellations |
| **Preprocessing** | `StandardScaler` for numeric features, `OneHotEncoder` (rare categories grouped) for categorical features, inside a single `Pipeline` / `ColumnTransformer` |
| **Modeling** | 3-fold stratified cross-validation across Logistic Regression, Random Forest and Gradient Boosting |
| **Tuning** | `RandomizedSearchCV` on the best model family (Gradient Boosting) |
| **Threshold Analysis** | Swept decision thresholds (0.30–0.70) using cross-validated predictions |
| **Interpretability** | Permutation feature importance measured on the held-out test set |
| **Deployment-ready** | Final model saved with `joblib` together with the chosen threshold and feature list |

---

## ⚠️ A Note on Data Leakage

This dataset contains a column called `reservation_status` (`Check-Out`, `Canceled`, `No-Show`) that maps **exactly** onto the target `is_canceled`. A model trained with it would score close to 100% while being useless for real prediction, because that information only exists *after* the outcome is known. It is removed here, along with `assigned_room_type`, so the reported results reflect what a hotel could actually predict at booking time.

---

## 📊 Key Results

### Model comparison (3-fold cross-validation on the training set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | `0.7974` | `0.6757` | `0.5075` | `0.5796` | `0.8540` |
| Random Forest | `0.8438` | `0.7756` | `0.6086` | `0.6821` | `0.9092` |
| Gradient Boosting | `0.8440` | `0.7450` | `0.6586` | `0.6991` | `0.9108` |

Gradient Boosting was selected and tuned (best parameters: `learning_rate=0.03, max_leaf_nodes=63, max_depth=None, min_samples_leaf=20, l2_regularization=1.0`), reaching a cross-validated ROC-AUC of `0.9139`.

### Final model on the held-out test set (17,446 bookings)

| Metric | Threshold 0.50 (default) | Threshold 0.35 (selected) |
|---|---|---|
| Accuracy | `0.8523` | `0.8429` |
| Precision (Canceled) | `0.7538` | `0.6794` |
| Recall (Canceled) | `0.6885` | `0.8132` |
| F1 (Canceled) | `0.7196` | `0.7403` |
| ROC-AUC | `0.9197` | — |

**Confusion matrix at the selected threshold (0.35):**

| | Predicted Not Canceled | Predicted Canceled |
|---|---|---|
| **Actual Not Canceled** | 10,801 | 1,843 |
| **Actual Canceled** | 897 | 3,905 |

**Interpretation:** lowering the threshold from 0.50 to 0.35 raises recall from 69% to 81%, so the model catches about 12 more cancellations out of every 100, at the cost of more false alarms (precision falls from 75% to 68%). For a hotel, missing a cancellation (an empty room nobody planned for) is typically costlier than flagging a booking that turns out to be fine, which is why the threshold was chosen on F1 from cross-validation rather than left at 0.50.

### What the data shows

- **Overall cancellation rate:** about 27.5% after removing duplicates.
- **City Hotel vs Resort Hotel:** 30.1% vs 23.5% cancellation.
- **Previous cancellations matter:** guests who canceled before cancel again 68% of the time, versus 26.7% for those who did not.
- **Special requests signal commitment:** bookings with no special requests cancel 33.3% of the time, falling steadily to 5.6% for bookings with five requests.
- **Non-refundable deposits:** 94.7% of non-refundable bookings are canceled. This looks counter-intuitive and is a known quirk of this dataset worth treating with caution rather than as a business insight.
- **Most influential features (permutation importance, drop in ROC-AUC):** `country` (0.129), `agent` (0.087), `required_car_parking_spaces` (0.079), `lead_time` (0.064), `market_segment` (0.060), `total_of_special_requests` (0.047).

---

## 🗂️ Project Structure

```
hotel-booking-cancellation-prediction/
│
├── data/
│   └── hotel_bookings.csv
│
├── notebooks/
│   └── hotel_cancellation_analysis.ipynb
│
├── models/
│   └── hotel_cancellation_model.pkl
│
├── visuals/
│   ├── cancellation_by_hotel.png
│   ├── cancellation_by_deposit_type.png
│   ├── lead_time_vs_cancellation.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── permutation_importance.png
│   └── ... (other charts)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Nahidhassn/hotel-booking-cancellation-prediction.git
cd hotel-booking-cancellation-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/hotel_cancellation_analysis.ipynb
```

> Note: the cross-validation and tuning steps train several models on ~70,000 rows, so a full run takes a few minutes.

---

## 🧠 Using the Trained Model

```python
import joblib
import pandas as pd

artifact = joblib.load("models/hotel_cancellation_model.pkl")
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

probability = model.predict_proba(new_booking_df[features])[:, 1]
prediction = (probability >= threshold).astype(int)  # 1 = likely to cancel
```

---

## 🤔 Why Gradient Boosting?

Logistic Regression was kept as a transparent baseline, but it cannot capture the non-linear interactions in this data (for example how lead time, country and market segment combine). Random Forest and Gradient Boosting both handle that, and Gradient Boosting gave the best ROC-AUC and recall in cross-validation while training faster than a large forest. Because accuracy alone hides the imbalance between kept and canceled bookings, the comparison relies on F1 and ROC-AUC, and the decision threshold is tuned explicitly.

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn
- **Machine Learning** — scikit-learn (Pipelines, ColumnTransformer, Random Forest, HistGradientBoosting, RandomizedSearchCV, permutation importance)
- **Model Persistence** — joblib

---

## 📌 Future Improvements

- Use a **time-based split** (train on earlier arrivals, test on later ones) to better mimic real deployment, since this project uses a random split
- Add gradient boosting libraries such as XGBoost or LightGBM and compare
- Add probability calibration so predicted probabilities can be read as true likelihoods
- Turn the model into a simple scoring tool that ranks upcoming bookings by cancellation risk

---

## 👤 Author

**Nahid Hasan**
[Email](mailto:nahidhassan.ctc@gmail.com)

*This project was built as part of my data science portfolio to demonstrate an end-to-end ML workflow: from raw data and leakage checks to a tuned, evaluated and deployable classification model.*
