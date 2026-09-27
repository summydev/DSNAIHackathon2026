# ==============================================================================
# DSN 2026 AI BOOTCAMP HACKATHON - FINAL SUBMISSION SCRIPT
# Author: Sumayah Adegbite
# Final Leaderboard Target: 1072.11
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

print("1. LOADING & CLEANING DATA")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
TARGET = "total_sales"
test_ids = test["id"].copy()

categorical_cols = [
    "product_code", "fat_content", "product_category",
    "store_code", "store_size", "store_location_tier", "store_format"
]

for col in categorical_cols:
    train[col] = train[col].fillna("unknown").astype(str)
    test[col] = test[col].fillna("unknown").astype(str)

global_weight_med = train["product_weight_kg"].median()
train["product_weight_kg"] = train["product_weight_kg"].fillna(global_weight_med)
test["product_weight_kg"] = test["product_weight_kg"].fillna(global_weight_med)

y = train[TARGET].values
y_log = np.log1p(y)
y_sqrt = np.sqrt(y)
EPS = 1e-6
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ==============================================================================
# PART 1: THE SCALPEL CORRECTOR (SAFE BASE MODEL)
# ==============================================================================
print("\n2. TRAINING PART 1: THE SCALPEL CORRECTOR")
X_tr_base = train.drop(columns=["id", TARGET]).copy()
X_te_base = test.drop(columns=["id"]).copy()

for col in categorical_cols:
    X_tr_base[col] = X_tr_base[col].astype("category")
    X_te_base[col] = X_te_base[col].astype("category")

TE_COLS_BASE = [
    ("product_code", 20), ("store_code", 10), ("product_category", 10),
    ("store_format", 10), ("store_location_tier", 10), ("store_size", 10)
]

def add_te(X_t, X_v, X_e, y_t_log, te_cols):
    X_t, X_v, X_e = X_t.copy(), X_v.copy(), X_e.copy()
    global_mean = np.mean(y_t_log)
    for col, smoothing in te_cols:
        temp = pd.DataFrame({col: X_t[col].astype(str).values, "target": y_t_log})
        stats = temp.groupby(col)["target"].agg(["mean", "count"])
        stats["encoding"] = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        X_t[f"{col}_te"] = X_t[col].astype(str).map(stats["encoding"]).fillna(global_mean)
        X_v[f"{col}_te"] = X_v[col].astype(str).map(stats["encoding"]).fillna(global_mean)
        X_e[f"{col}_te"] = X_e[col].astype(str).map(stats["encoding"]).fillna(global_mean)
    return X_t, X_v, X_e

oof_cat_base = np.zeros(len(X_tr_base))
oof_lgb_base = np.zeros(len(X_tr_base))
test_cat_base, test_lgb_base = [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_base), 1):
    X_t, X_v, X_e = add_te(X_tr_base.iloc[tr_idx], X_tr_base.iloc[val_idx], X_te_base, y_log[tr_idx], TE_COLS_BASE)
    cat_idx = [X_t.columns.get_loc(c) for c in categorical_cols]
    
    cat = CatBoostRegressor(depth=6, iterations=3000, learning_rate=0.03, l2_leaf_reg=3, random_seed=42, verbose=False)
    cat.fit(X_t, y_log[tr_idx], cat_features=cat_idx, eval_set=(X_v, y_log[val_idx]), use_best_model=True, verbose=False)
    
    c_v = np.expm1(cat.predict(X_v))
    corr_cat = np.mean(y[tr_idx]) / (np.mean(np.maximum(c_v, 0)) + EPS)
    oof_cat_base[val_idx] = c_v * corr_cat
    test_cat_base.append(np.expm1(cat.predict(X_e)) * corr_cat)
    
    lgb_m = LGBMRegressor(objective="regression", learning_rate=0.03, num_leaves=64, n_estimators=5000, random_state=42, verbosity=-1)
    lgb_m.fit(X_t, y[tr_idx], eval_set=[(X_v, y[val_idx])], callbacks=[early_stopping(200, verbose=False)])
    
    l_v = np.maximum(lgb_m.predict(X_v), 0)
    corr_lgb = np.mean(y[tr_idx]) / (np.mean(l_v) + EPS)
    oof_lgb_base[val_idx] = l_v * corr_lgb
    test_lgb_base.append(np.maximum(lgb_m.predict(X_e), 0) * corr_lgb)

oof_blend_base = 0.63 * oof_cat_base + 0.37 * oof_lgb_base
test_blend_base = 0.63 * np.mean(test_cat_base, axis=0) + 0.37 * np.mean(test_lgb_base, axis=0)

# Scalpel Corrector (FIXED BUG)
train_res = train.copy()
test_res = test.copy()
train_res["base_pred"] = oof_blend_base
test_res["base_pred"] = test_blend_base

res_features = ["store_code", "store_format", "product_category", "product_price", "base_pred"]
X_res_tr = train_res[res_features].copy()
X_res_te = test_res[res_features].copy()

X_res_tr["clean_category"] = X_res_tr["product_category"].str.lower().str.strip()
X_res_te["clean_category"] = X_res_te["product_category"].str.lower().str.strip()

for col in ["store_code", "store_format", "clean_category"]:
    X_res_tr[col] = X_res_tr[col].astype("category")
    X_res_te[col] = X_res_te[col].astype("category")

y_res = y - oof_blend_base
oof_scalpel = np.zeros(len(X_res_tr))
test_scalpel = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_res_tr), 1):
    model = LGBMRegressor(objective="regression", learning_rate=0.02, num_leaves=12, max_depth=4, n_estimators=1500, random_state=42, verbosity=-1)
    model.fit(X_res_tr.iloc[tr_idx], y_res[tr_idx], eval_set=[(X_res_tr.iloc[val_idx], y_res[val_idx])], callbacks=[early_stopping(100, verbose=False)])
    test_scalpel.append(model.predict(X_res_te))

final_test_scalpel = np.maximum(test_blend_base + np.mean(test_scalpel, axis=0), 0)
pd.DataFrame({"id": test_ids, TARGET: final_test_scalpel}).to_csv("submission_scalpel_corrector.csv", index=False)
print("✅ Scalpel File Generated: submission_scalpel_corrector.csv")

# ==============================================================================
# PART 2: THE INTERACTION TRINITY (AGGRESSIVE LB MODEL)
# ==============================================================================
print("\n3. TRAINING PART 2: THE INTERACTION TRINITY")
X_tr_int = train.drop(columns=["id", TARGET]).copy()
X_te_int = test.drop(columns=["id"]).copy()

X_tr_int["prod_store"] = X_tr_int["product_code"] + "_" + X_tr_int["store_code"]
X_te_int["prod_store"] = X_te_int["product_code"] + "_" + X_te_int["store_code"]
X_tr_int["prod_format"] = X_tr_int["product_code"] + "_" + X_tr_int["store_format"]
X_te_int["prod_format"] = X_te_int["product_code"] + "_" + X_te_int["store_format"]

cat_cols_ext = categorical_cols + ["prod_store", "prod_format"]
for col in cat_cols_ext:
    X_tr_int[col] = X_tr_int[col].astype("category")
    X_te_int[col] = X_te_int[col].astype("category")

TE_COLS_INT = TE_COLS_BASE + [("prod_store", 50), ("prod_format", 30)]

test_cat_int, test_lgb_int, test_xgb_int = [], [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_int), 1):
    X_t, X_v, X_e = add_te(X_tr_int.iloc[tr_idx], X_tr_int.iloc[val_idx], X_te_int, y_log[tr_idx], TE_COLS_INT)
    cat_idx = [X_t.columns.get_loc(c) for c in cat_cols_ext]
    
    # Pillar 1: CatBoost
    cat = CatBoostRegressor(depth=6, iterations=3000, learning_rate=0.03, random_seed=42, verbose=False)
    cat.fit(X_t, y_log[tr_idx], cat_features=cat_idx, eval_set=(X_v, y_log[val_idx]), use_best_model=True, verbose=False)
    test_cat_int.append(np.expm1(cat.predict(X_e)))
    
    # Pillar 2: LightGBM
    lgb_m = LGBMRegressor(objective="regression", learning_rate=0.03, num_leaves=64, n_estimators=5000, random_state=42, verbosity=-1)
    lgb_m.fit(X_t, y[tr_idx], eval_set=[(X_v, y[val_idx])], callbacks=[early_stopping(200, verbose=False)])
    test_lgb_int.append(np.maximum(lgb_m.predict(X_e), 0))

    # Pillar 3: XGBoost (Sqrt space)
    X_t_x, X_v_x, X_e_x = X_t.drop(columns=cat_cols_ext), X_v.drop(columns=cat_cols_ext), X_e.drop(columns=cat_cols_ext)
    xgb_m = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.03, max_depth=6, n_estimators=3000, random_state=42)
    xgb_m.fit(X_t_x, y_sqrt[tr_idx], eval_set=[(X_v_x, y_sqrt[val_idx])], verbose=False)
    test_xgb_int.append(np.maximum(xgb_m.predict(X_e_x), 0) ** 2)

test_blend_int = (np.mean(test_cat_int, axis=0) + np.mean(test_lgb_int, axis=0) + np.mean(test_xgb_int, axis=0)) / 3.0

# ==============================================================================
# PART 3: THE GOLDEN BLEND
# ==============================================================================
print("\n4. CREATING THE GOLDEN BLEND")
w_agg = 0.65
w_con = 0.35

blended_sales = (test_blend_int * w_agg) + (final_test_scalpel * w_con)
blended_sales = np.maximum(blended_sales, 0)

pd.DataFrame({"id": test_ids, TARGET: blended_sales}).to_csv("submission_golden_blend.csv", index=False)
print("✅ Golden Blend File Generated: submission_golden_blend.csv")
print("\n🎉 DONE! Submit your files and upload this script to your GitHub.")
