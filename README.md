# Part 2 – Machine Learning Model Development

## Project Overview

This part of the project develops and evaluates both regression and classification models using the cleaned dataset produced in Part 1. The workflow includes data preprocessing, feature encoding, train-test splitting, feature scaling, regression modeling, classification modeling, model evaluation, threshold analysis, regularization experiments, and bootstrap confidence interval analysis.

---

## Dataset

**Input File:** `cleaned_data.csv`

The dataset contains information about students' AI tool usage, study habits, awareness, and academic impact.

---

## Target Definitions

### Regression Target (`y_reg`)

The regression target is **Daily_Usage_Hours**, which is a continuous numerical variable representing the number of hours students use AI tools daily.

### Classification Target (`y_clf`)

A binary classification label was created from the regression target using the median value.

```python
y_clf = (y_reg > y_reg.median()).astype(int)
```

* Class **1** → Daily_Usage_Hours greater than the median
* Class **0** → Daily_Usage_Hours less than or equal to the median

---

# Feature Encoding

Categorical variables were encoded according to their characteristics.

## Label Encoding

The **Internet_Access** feature has a natural order.

| Category | Encoded Value |
| -------- | ------------: |
| Poor     |             0 |
| Medium   |             1 |
| High     |             2 |

This preserves the ordinal relationship between the categories.

## One-Hot Encoding

The following features have no natural ordering and were one-hot encoded using `pd.get_dummies(drop_first=True)`.

* Student_Name
* College_Name
* Stream
* AI_Tools_Used
* Use_Cases
* Do_Professors_Allow_Use
* Preferred_AI_Tool
* Willing_to_Pay_for_Access
* State
* Device_Used

The first dummy column was removed to avoid multicollinearity.

One-hot encoding prevents the model from assuming a false ordinal relationship between categories that do not have any meaningful order.

---

# Train-Test Split and Feature Scaling

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

using:

* `test_size = 0.20`
* `random_state = 42`

Feature scaling was performed using **StandardScaler**.

The scaler was fitted **only on the training data** and then applied to both training and testing datasets.

Fitting the scaler on the entire dataset would cause **data leakage**, because information from the test set (its mean and standard deviation) would influence the training process, resulting in overly optimistic performance estimates.

---

# Linear Regression

A Linear Regression model was trained using the standardized training data.

Evaluation metrics:

* Mean Squared Error (MSE)
* R² Score

### Coefficient Interpretation

Each coefficient represents the expected change in the predicted target value for a one-unit increase in the standardized feature while keeping all other features constant.

* A positive coefficient indicates that increasing the feature increases the predicted value.
* A negative coefficient indicates that increasing the feature decreases the predicted value.

The three features with the largest absolute coefficients were identified as the most influential predictors.

---

# Ridge Regression

A Ridge Regression model with **alpha = 1.0** was trained using the same training and testing split.

## Model Comparison

| Model             |                  MSE |                   R² |
| ----------------- | -------------------: | -------------------: |
| Linear Regression | *(Fill your output)* | *(Fill your output)* |
| Ridge Regression  | *(Fill your output)* | *(Fill your output)* |

### Why Ridge Regression?

Ridge Regression applies **L2 Regularization**, which reduces excessively large coefficients.

The **alpha** parameter controls the amount of regularization.

* Small alpha → Less regularization
* Large alpha → Stronger regularization

Regularization can reduce overfitting and improve model generalization.

---

# Logistic Regression

A Logistic Regression classifier was trained to predict the binary target variable.

## Class Imbalance

The training class distribution was checked using:

```python
value_counts()
```

If the minority class contained fewer than 35% of samples, **SMOTE** was applied only to the training dataset to balance the classes.

Otherwise, no resampling was required.

---

# Classification Evaluation

The following evaluation metrics were computed:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-score
* ROC Curve
* Area Under Curve (AUC)

---

# Precision

Precision = TP / (TP + FP)

where:

* TP = True Positives
* FP = False Positives

---

# Recall

Recall = TP / (TP + FN)

where:

* TP = True Positives
* FN = False Negatives

For this student AI usage dataset, **Recall** is generally more important if identifying all positive cases is the priority. If minimizing false positives is more important, Precision should be prioritized.

---

# ROC Curve

The ROC curve illustrates the trade-off between the True Positive Rate and False Positive Rate at different decision thresholds.

The Area Under the Curve (AUC) summarizes this performance.

* AUC = 0.50 → Random classifier
* AUC = 1.00 → Perfect classifier

Higher AUC values indicate better class separation.

---

# Decision Threshold Sensitivity

The default decision threshold (0.50) was varied from:

* 0.30
* 0.40
* 0.50
* 0.60
* 0.70

For each threshold, Precision, Recall, and F1-score were calculated.

The threshold with the highest F1-score was selected as the best operating threshold.

Lower thresholds generally increase Recall but reduce Precision.

Higher thresholds generally increase Precision but reduce Recall.

The appropriate threshold depends on the application requirements.

---

# Logistic Regression Regularization

A second Logistic Regression model was trained with:

**C = 0.01**

The baseline model used:

**C = 1.0**

## Comparison

| Model                        |            Precision |               Recall |                  AUC |
| ---------------------------- | -------------------: | -------------------: | -------------------: |
| Logistic Regression (C=1.0)  | *(Fill your output)* | *(Fill your output)* | *(Fill your output)* |
| Logistic Regression (C=0.01) | *(Fill your output)* | *(Fill your output)* | *(Fill your output)* |

The **C** parameter controls the inverse strength of regularization.

* Larger C → Less regularization
* Smaller C → Stronger regularization

Reducing C may improve generalization by reducing overfitting, although excessive regularization can decrease predictive performance.

---

# Bootstrap Confidence Interval

To evaluate whether the baseline Logistic Regression consistently outperformed the strongly regularized model, **500 bootstrap samples** were generated from the test set.

For each bootstrap sample:

* AUC for C = 1.0 was calculated.
* AUC for C = 0.01 was calculated.
* The AUC difference was recorded.

The following statistics were reported:

* Mean AUC Difference: *(Fill your output)*
* 2.5th Percentile: *(Fill your output)*
* 97.5th Percentile: *(Fill your output)*

If the 95% confidence interval excludes zero, the baseline model consistently outperforms the regularized model.

If the interval includes zero, the observed difference may not be statistically reliable.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib

---

# Conclusion

This project successfully developed regression and classification models using a cleaned dataset. Linear Regression, Ridge Regression, and Logistic Regression models were evaluated using appropriate performance metrics. Threshold sensitivity analysis, regularization experiments, and bootstrap confidence interval analysis were performed to ensure robust model evaluation and reliable performance assessment.
