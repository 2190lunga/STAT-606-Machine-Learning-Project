# STAT606 Assignment - Divorce Prediction Using Machine Learning

## Authors
- Lungani Zungu (219024060)
- Ntobeko Hlongwa (219021668)

---

## 1. Introduction to the Data and Problem

**Dataset:** Divorce Prediction Dataset  
**Source:** Kaggle  
**Observations:** 5000  
**Variables:** 22 (21 predictors + 1 target)  
**Objective:** Predict whether a married couple will divorce (binary: 0 = No, 1 = Yes) based on demographic, financial, emotional, and behavioural characteristics of their relationship.  
**Tools:** R 4.4.3, H2O 3.44.0.3, RStudio  
**Libraries:** dplyr, caTools, caret, pROC, h2o, rpart, rpart.plot  
**Split:** 70% training (3500 obs) / 30% test (1500 obs), seed = 606  
**Cross-validation:** 5-fold CV used during model training

---

## 2. Data Cleaning (What We Found and Fixed)

**Issue 1:** CSV file used semicolons (;) as separators instead of commas.  
- **Fix:** Converted all semicolons to commas before loading into R.  
- **Why:** R's read.csv() expects commas by default.

**Issue 2:** Trailing empty columns and "#VALUE!" errors at the end of each row (Excel export issue).  
- **Fix:** Removed trailing commas and #VALUE! entries from every row.  
- **Why:** These were junk - not real data. Would cause errors in modelling.

**Issue 3:** Five text (character) columns needed converting to factors for modelling.  
- **Fix:** Used mutate(across(where(is.character), as.factor)) to convert them.  
- **Why:** Models require categorical variables as factors, not free text.

**Issue 4:** Six binary columns (0/1) were treated as numeric by R.  
- **Fix:** Manually converted each to factor (e.g., factor(df$mental_health_issues)).  
- **Why:** These are categories (Yes/No), not continuous numbers.

**No issues found:**  
- No ID column present - no removal needed.  
- No missing values in any column (all zeros from colSums(is.na(df))).  
- No high cardinality - highest is education_level with only 5 levels.  
- No extremely sparse categories requiring removal or collapsing.

---

## 3. Basic Characteristics of the Data

- **Number of observations:** 5000
- **Number of attributes:** 21 predictors + 1 target = 22 columns
- **Target variable:** `divorced` (0 = not divorced, 1 = divorced)
- **Class balance:** 60.18% not divorced (3009), 39.82% divorced (1991) - no balancing required
- **ID variable:** None present
- **Missing values:** None

### Variable Types:
| Type | Variables |
|------|-----------|
| Numeric continuous | age_at_marriage, marriage_duration_years, combined_income, communication_score, financial_stress_level, social_support, trust_score |
| Numeric discrete | num_children, conflict_frequency, shared_hobbies_count |
| Categorical | education_level (5 levels), employment_status (4), religious_compatibility (3), conflict_resolution_style (4), marriage_type (3) |
| Binary (0/1) | cultural_background_match, mental_health_issues, infidelity_occurred, counseling_attended, pre_marital_cohabitation, domestic_violence_history |

---

## 4. Models Fitted (70:30 Train/Test Split)

| # | Model | Algorithm | Library | Hyperparameter Tuning |
|---|-------|-----------|---------|----------------------|
| 1 | Naive Bayes | h2o.naiveBayes | h2o | laplace = 0, 5-fold CV |
| 2 | Decision Tree (H2O) | h2o.grid (GBM ntrees=1) → h2o.decision_tree | h2o | Grid search over max_depth (3–21) and min_rows (1,5,10,20,50). Best: max_depth=7, min_rows=20 |
| 3 | Decision Tree (rpart) | rpart | rpart | maxdepth = 4, 5-fold CV |
| 4 | Logistic Regression | h2o.glm | h2o | family = "binomial", lambda = 0, compute_p_values = TRUE |

---

## 5. Model Comparison & Results

**Metric chosen:** AUC (Area Under the ROC Curve) - justified because it measures overall ranking ability regardless of threshold choice, and is not affected by class imbalance.

| Model | Train AUC | Test AUC | Train F1 | Test F1 | Overfitting? |
|-------|-----------|----------|----------|---------|--------------|
| Naive Bayes | 0.6155 | 0.5663 | 0.3047 | 0.2369 | No - but underfitting (poor on both sets) |
| Decision Tree (H2O) | 0.7076 | 0.5170 | 0.4728 | 0.3186 | **Yes** - large train-test gap (0.19 AUC drop) |
| Decision Tree (rpart) | 0.5195 | 0.5139 | 0.1301 | 0.1215 | No - severe underfitting (only 1 split) |
| Logistic Regression | 0.6121 | 0.5711 | 0.2962 | 0.2506 | No - good generalisation, but weak overall |

**Best model:** Logistic Regression (Test AUC = 0.5711)  
**Reason:** Highest test AUC among all four models, and the smallest train-test gap indicating good generalisation. However, all models performed poorly on this dataset - none achieved strong predictive power.

---

## 6. Overfitting / Underfitting Comments

**Naive Bayes:** Train AUC (0.62) ≈ Test AUC (0.57) - no overfitting, but the model underfits. It performs barely better than random guessing (AUC = 0.5). The "naive" independence assumption likely does not hold for this data - predictors like trust_score and communication_score are likely correlated, which violates the assumption.

**Decision Tree (H2O):** Train AUC (0.71) >> Test AUC (0.52) - **overfitting**. The model learned the training data too well (high train performance) but failed to generalise to unseen data (large 0.19 AUC drop). The grid search selected max_depth = 7 with min_rows = 20, but the tree still memorised noise in the training set. This is a classic sign of overfitting - the model captures patterns that exist only in the training data, not the true underlying relationship.

**Decision Tree (rpart):** Train AUC (0.52) ≈ Test AUC (0.51) - no overfitting, but severe underfitting. The tree only used 1 variable (domestic_violence_history), meaning it could not find useful splits in any other feature. This is effectively a stump, not a real tree. The complexity parameter (CP) pruned away all other splits because they did not improve classification enough to justify the added complexity.

**Logistic Regression:** Train AUC (0.61) ≈ Test AUC (0.57) - no overfitting. The small gap shows good generalisation, but the model underfits. It found 6 statistically significant predictors (p < 0.05), yet none were strong enough individually to produce high discriminative power.

---

## 7. Top 5 Most Important Features

Based on Logistic Regression p-values:

| Rank | Variable | p-value | Odds Ratio | Interpretation |
|------|----------|---------|------------|----------------|
| 1 | domestic_violence_history | 0.0000 | 2.23 | History of domestic violence more than doubles the odds of divorce |
| 2 | infidelity_occurred | 0.0000 | 1.57 | Infidelity increases divorce odds by 57% |
| 3 | financial_stress_level | 0.0000 | 1.08 | Each 1-unit increase in financial stress raises divorce odds by 8% |
| 4 | communication_score | 0.0000 | 0.91 | Higher communication score reduces divorce odds by 9% per unit (protective) |
| 5 | trust_score | 0.0166 | 0.96 | Higher trust reduces divorce odds by 4% per unit (protective) |

**6th significant variable:** mental_health_issues (p = 0.0250, OR = 1.21) - having mental health issues increases divorce odds by 21%.

**Key insight:** The two strongest predictors are binary events (domestic violence and infidelity), while continuous relationship quality scores (communication, trust) are protective but weaker. Financial stress is a moderate risk factor.

---

## 8. Conclusion

- **Best performing model:** Logistic Regression (Test AUC = 0.5711) - best generalisation and interpretability.
- **Worst performing model:** Decision Tree rpart (Test AUC = 0.5139) - essentially random guessing.
- **Overall finding:** All four models struggled to predict divorce from these features (all AUCs between 0.51–0.57). This suggests the 21 predictors in this dataset do not contain strong enough signals to reliably distinguish divorced from non-divorced couples. The features may be too noisy, or critical predictors (e.g., emotional intimacy, relationship satisfaction scales) may be missing from the dataset.
- **Overfitting was observed** only in the H2O Decision Tree, which had the highest training AUC (0.71) but collapsed on the test set (0.52).
- **No data balancing was needed** - the class split (60/40) was close enough to not require SMOTE or undersampling.
