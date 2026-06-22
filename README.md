# Loan Approval Prediction

A machine learning project that predicts loan approval status using a Decision Tree Classifier, with a focus on handling class imbalance and threshold tuning for a business-relevant precision-recall trade-off.

## Overview

This project applies binary classification to predict whether a loan application will be approved or rejected, based on applicant financial and credit information.

## Dataset

- Rows: 45,000 loan applications
- Features: 13 (demographic, financial, and credit-related)
- Target: Loan Status (1 = Approved, 0 = Rejected)
- Class balance: 77.8% Rejected, 22.2% Approved (significant imbalance)

## Workflow

1. Exploratory Data Analysis
   - Identified significant class imbalance in the target variable
   - Found Person Income to be heavily right-skewed with extreme outliers (max value over 7 million against a median of about 67,000)
   - Capped Person Income at the 99th percentile to reduce the influence of extreme outliers

2. Feature Encoding
   - Binary columns (Gender, Previous Loan) mapped directly to 0/1
   - Education encoded as ordinal, since degree levels follow a natural order (High School to Doctorate)
   - Remaining nominal columns (Home Ownership, Loan Intent) encoded using One-Hot Encoding

3. Modeling
   - Trained a Decision Tree Classifier with class_weight="balanced" to address class imbalance
   - Used stratified train-test split to preserve class distribution across train and test sets

4. Threshold Tuning (Key Decision)
   - Default threshold (0.5): Precision = 0.60, Recall = 0.94 for the Approved class
   - Increased threshold to 0.7: Precision improved to 0.84, Recall dropped to 0.75
   - In a lending context, approving a loan that should have been rejected (false positive) carries direct financial risk, so prioritizing precision was the more appropriate business decision

5. Critical Analysis of Feature Importance
   - The Previous Loan feature dominated the model's decisions, accounting for 70% of total feature importance
   - A cross-tabulation revealed that when Previous Loan = Yes, the rejection rate was 100%, with zero exceptions
   - This level of determinism is unusual for real-world financial data and suggests the dataset may contain a synthetic or overly clean pattern rather than a fully realistic relationship; this is flagged here as a limitation rather than treated as a reliable real-world insight

## Results

| Metric (Approved class) | Threshold 0.5 | Threshold 0.7 |
|---|---|---|
| Precision | 0.60 | 0.84 |
| Recall | 0.94 | 0.75 |
| F1-Score | 0.73 | 0.80 |

ROC-AUC Score: 0.960

## Key Takeaways

- Tree-based models like Decision Trees are largely unaffected by outliers and feature scale, unlike linear models
- Threshold tuning should be guided by the cost of different error types in the specific business context, not chosen by default
- Extremely strong or deterministic feature relationships should be investigated rather than accepted at face value, as they may indicate data artifacts rather than genuine patterns

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (DecisionTreeClassifier, train_test_split, classification_report, roc_auc_score)
- Matplotlib, Seaborn

## Files

- 04_loan_approval_prediction.ipynb: full analysis and modeling notebook

## How to Run

1. Clone this repository
2. Open 04_loan_approval_prediction.ipynb in Jupyter Notebook or Google Colab
3. Download the dataset from Kaggle and place the CSV file in the same directory
4. Run all cells
