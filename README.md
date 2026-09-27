# DSNAIHackathon2026
# DSN 2026 ML Track — Predict Sale

Final public LB: 1072.12 (rank ~15/285, top 5% of honest submissions)

## Approach
- Data understanding: verified (product × store) matrix structure,
  variance decomposition (store 66%, product 58%, interaction 4%)
- Feature engineering: structural missingness handling, target encoding
  inside folds, product profile aggregates, cell-vs-product residuals
- Base models: CatBoost (log target) + LGBM (raw target)
- Corrections: bias correction, residual correction model, blend search
- Final: multi-stage blend at OOF 1135 → LB 1072

## Notebook structure
1. Data loading and inspection
2. EDA (missingness, cardinality, matrix structure)
3. Cleaning and feature engineering
4. Cross-validation with in-fold target encoding
5. Base model training
6. Blend and residual correction
7. Submission generation
