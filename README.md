# DSN 2026 AI Bootcamp Hackathon: Retail Sales Forecasting

This repository contains the machine learning pipeline developed for the DSN 2026 AI Bootcamp Hackathon. The objective of this project is to accurately predict `total_sales` for various retail items across different store locations using historical sales data.

## 🏆 Results
* **Best Cross-Validation (CV) Score:** 1095.82 RMSE
* **Public Leaderboard Score:** 1072.11 RMSE

## 🧠 Methodology & Architecture

The final solution relies on a robust ensemble of two distinct machine learning pipelines, balancing conservative generalization with aggressive local pattern recognition.

### 1. The "Scalpel Corrector" (Conservative Pipeline)
Designed to prioritize Cross-Validation stability and prevent overfitting on sparse features.
* **Base Models:** A blend of CatBoost (predicting in Log space) and LightGBM (predicting in Raw space).
* **Feature Engineering:** Utilizes smoothed Target Encoding on clean categorical features (`store_code`, `product_category`, `store_format`).
* **Residual Modeling (The Scalpel):** A secondary LightGBM model trained strictly on the residuals (errors) of the base blend, using `product_price` and store characteristics to surgically correct massive underpredictions on high-volume items ("whales") without exploding the variance.

### 2. The "Interaction Trinity" (Aggressive Pipeline)
Designed to exploit specific data leakage and spatial mapping between the training and test sets.
* **Feature Engineering:** Extracts highly specific interaction features (`product_code` + `store_code` and `product_code` + `store_format`) mapped via heavy target encoding.
* **The Trinity Ensemble:** 
  * CatBoost Regressor (Logarithmic target space)
  * LightGBM Regressor (Raw target space)
  * XGBoost Regressor (Square Root target space for variance stabilization)

### 3. The Golden Blend
The final submission is a weighted average combining the uncorrelated error profiles of both pipelines:
* **65%** Interaction Trinity
* **35%** Scalpel Corrector

## ⚙️ How to Run

1. Clone this repository.
2. Ensure you have the required dependencies installed:
   ```bash
   pip install pandas numpy scikit-learn catboost lightgbm xgboost
