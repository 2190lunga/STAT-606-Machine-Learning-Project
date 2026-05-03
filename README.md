# STAT606 Assignment - Divorce Prediction Using Machine Learning

## Authors
- Lungani Zungu (219024060)
- Ntobeko Hlongwa (219021668)

---

## 1. Introduction to the Data and Problem

**Dataset:** Divorce Prediction Dataset  
**Source:** [fill in - Kaggle/UCI/etc.]  
**Observations:** 5000  
**Variables:** 22 (21 predictors + 1 target)  
**Objective:** Predict whether a married couple will divorce (binary: 0 = No, 1 = Yes) based on characteristics of their relationship and background.

---

## 2. Data Cleaning (What We Found and Fixed)

**Issue 1:** CSV file used semicolons (;) as separators instead of commas.  
- **Fix:** Converted all semicolons to commas before loading into R.  
- **Why:** R's read.csv() expects commas by default.

**Issue 2:** Trailing empty columns and "#VALUE!" errors at the end of each row (Excel export issue).  
- **Fix:** Removed trailing commas and #VALUE! entries from every row.  
- **Why:** These were junk — not real data. Would cause errors in modelling.

**Issue 3:** Five text (character) columns needed converting to factors for modelling.  
- **Fix:** Used mutate(across(where(is.character), as.factor)) to convert them.  
- **Why:** Models require categorical variables as factors, not free text.

**Issue 4:** Six binary columns (0/1) were treated as numeric by R.  
- **Fix:** Manually converted each to factor (e.g., factor(df$mental_health_issues)).  
- **Why:** These are categories (Yes/No), not continuous numbers.

**No issues found:**  
- No ID column present — no removal needed.  
- No missing values in any column (all zeros from colSums(is.na(df))).  
- No high cardinality — highest is education_level with only 5 levels.  
- No extremely sparse categories requiring removal or collapsing.

---

## 3. Basic Characteristics of the Data

- **Number of observations:** 5000
- **Number of attributes:** 21 predictors + 1 target = 22 columns
- **Target variable:** `divorced` (0 = not divorced, 1 = divorced)
- **Class balance:** 60.18% not divorced (3009), 39.82% divorced (1991) — no balancing required
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

1. Naive Bayes
2. Decision Tree (H2O - pre-pruning)
3. Decision Tree (rpart - post-pruning)
4. Logistic Regression

---

## 5. Model Comparison & Results

**Metric chosen:** AUC (Area Under the ROC Curve) — justified because it measures overall ranking ability regardless of threshold choice, and is not affected by class imbalance.

| Model | Train AUC | Test AUC | Overfitting? |
|-------|-----------|----------|--------------|
| Naive Bayes | 0.6155 | 0.5663 | No — but underfitting (poor performance on both sets) |
| Decision Tree (H2O) |0.7076 |0.517 | Yes - performing well on the train set but poorly on the test set | 
| Decision Tree (rpart) |0.5195 |0.5139 |No - but underfitting |
| Logistic Regression | | | |

**Best model:** [to be determined after all models are fitted]  
**Reason:** [to be determined]

---

## 6. Overfitting / Underfitting Comments

**Naive Bayes:** Train AUC (0.62) ≈ Test AUC (0.57) — no overfitting, but the model underfits. It performs barely better than random guessing (AUC = 0.5). The "naive" independence assumption likely does not hold for this data — predictors like trust_score and communication_score are likely correlated, which violates the assumption.

**Decision Tree (H2O):** Train AUC (0.71) ~ Test AUC(0.52) - the model is overfitting. 
**Decision Tree (rpart):** Train AUC (0.52) ~ Test AUC(0.51) -
**Logistic Regression :** Train AUC () ~ Test AUC() -

[Remaining models to be assessed]

---

## 7. Top 5 Most Important Features

Based on Logistic Regression p-values:

| Rank | Variable | p-value | Odds Ratio | Interpretation |
|------|----------|---------|------------|----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
