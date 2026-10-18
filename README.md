Early identification of at-risk students on OULAD: a comparison of machine learning methods


The script compares four classifiers — logistic regression, decision tree, random forest and gradient boosting — on the Open University Learning Analytics Dataset (OULAD). The task is binary classification of a student × module × presentation record:

Class	Final result	Meaning
1	Pass, Distinction	success
0	Fail, Withdrawn	at-risk group

Everything is implemented in a single, reproducible script (oulad.py) built on scikit-learn.

Key design decisions
No target leakage. Final exam results are excluded from the features, as are final_result and date_unregistration (the latter is used only to decide who is still enrolled at a window boundary).
Observation windows. Features are built from events up to day 30, 60 or 90 of the presentation (W30, W60, W90) and for the whole course (FULL). In the early windows, students who had already unregistered by the cutoff are removed. W30–W90 model early warning; FULL is a retrospective control scenario.
Student-level split. One student can appear in several records, so the data are split by id_student with StratifiedGroupKFold (the first fold, about 20 % of students, is the test set). The same test students are used for every window.
Leak-free pipelines. Imputation, scaling (logistic regression only) and one-hot encoding live inside a scikit-learn Pipeline and are fitted on training data only, including inside every cross-validation fold.
Model selection by cross-validation, not by the test set. The best model per window is chosen by mean macro-F1 over repeated grouped 5-fold CV (3 repeats, 15 folds).
Fixed hyperparameters by default. An optional RandomizedSearchCV mode exists (TUNE=1) but was not used for the reported results.
What the script does
Loads the CSV files with na_values=["?"] and reports genuine missing values.
Builds 16 features (45 after encoding) in four groups: demographics, course, VLE activity, assessments.
Trains the four models for each window and evaluates them with repeated grouped CV and on the held-out test set (accuracy, ROC-AUC, per-class precision/recall/F1, macro-F1, confusion matrix).
Computes 95 % bootstrap confidence intervals (resampling test records).
Tests the best model against the others: Nadeau–Bengio corrected resampled t-test (CV) and exact McNemar test (test set).
Builds a threshold table for the at-risk class (recall targets 0.80 / 0.85 / 0.90 / 0.95).
Runs a feature-group ablation (windows W60 and FULL).
Computes permutation importance and impurity-based importance for the best model.
Saves all fitted pipelines with joblib and checks that reloaded models reproduce the predictions.
Features
Group	Features
Demographics	gender, region, highest_education, imd_band, age_band, disability, num_of_prev_attempts, studied_credits
Course	code_module, date_registration
VLE activity	total_clicks, active_days
Assessments	mean_score, n_assessments, share_submitted, has_submissions
Error terminology

Class 1 (success) is the positive class. Therefore FP is an at-risk student predicted as successful (a missed risk), and FN is a successful student flagged as at risk (a false alarm).

Requirements

Python 3 and the packages below. The reported results were produced with the versions in parentheses (see results/environment.json in a finished run):

pip install numpy pandas scipy scikit-learn joblib matplotlib seaborn

Python 3.14.8, pandas 3.0.6, NumPy 2.5.3, scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0, matplotlib 3.11.2, seaborn 0.13.2.

Data

The dataset is not included in this repository. Download OULAD from the official page and place these files in one directory:

courses.csv
studentInfo.csv
studentRegistration.csv
assessments.csv
studentAssessment.csv
studentVle.csv

(vle.csv is not used.)

Usage
bash
# full run (default)
OULAD_DIR=/path/to/oulad python oulad.py

# quick smoke test (1 CV repeat, 200 bootstrap samples)
QUICK=1 OULAD_DIR=/path/to/oulad python oulad.py

# optional hyperparameter search on the training part
TUNE=1 OULAD_DIR=/path/to/oulad python oulad.py

On Windows PowerShell, set variables with $env:OULAD_DIR="C:\path\to\oulad"; python oulad.py.

Environment variable	Default	Description
OULAD_DIR	.	Directory with the OULAD CSV files
OUT_DIR	results	Output directory
QUICK	0	1 = fast run (1 CV repeat, 200 bootstrap samples)
TUNE	0	1 = RandomizedSearchCV on the training set
TUNE_ITER	15	Number of random-search iterations when TUNE=1

Random seed: RANDOM_STATE = 42.

Default hyperparameters
Model	Settings
Logistic regression	max_iter=1000, C=1 (L2), standardized numeric features
Decision tree	max_depth=10
Random forest	n_estimators=100, max_depth=12
Gradient boosting	n_estimators=100, learning_rate=0.1, max_depth=3
Output files (results/)
File	Content
environment.json	Python/library versions and run settings
tables_and_missing.csv	Table sizes and missing values
dataset_summary.csv	Records, students, train/test sizes and class counts per window
feature_list.csv	Feature names, groups and types
results_holdout.csv	Test-set metrics and confusion matrices
results_cv.csv, cv_folds_raw.csv	Cross-validation summary and per-fold results
bootstrap_ci.csv	95 % bootstrap confidence intervals
significance_tests.csv	Nadeau–Bengio and McNemar test results
threshold_table_risk.csv	Risk-threshold analysis
ablation.csv	Feature-group ablation
importance_permutation.csv, importance_impurity.csv	Feature importance
best_params.json	Tuned parameters (empty unless TUNE=1)
confusion_*.png, roc_*.png, risk_precision_recall.png, macro_f1_by_window.png	Figures
models/<window>/<Model>.joblib, models/<window>/metadata.json, saved_models.csv	Saved pipelines and reload check
Main results

Macro-F1 on the held-out test set (threshold 0.5):

Model	W30	W60	W90	FULL
Logistic regression	0.6961	0.7431	0.7629	0.9150
Decision tree	0.6829	0.7272	0.7587	0.9111
Random forest	0.7060	0.7445	0.7685	0.9167
Gradient boosting	0.7063	0.7508	0.7753	0.9194
Gradient boosting has the highest cross-validated macro-F1 in every window, but its advantage over logistic regression is small (0.005–0.014 in CV) and over random forest it is not statistically distinguishable on the test set.
Assessment features (mean_score, share_submitted) contribute most; VLE activity adds little once assessment results are available; demographic features contribute very little.
With the default threshold of 0.5, recall for the at-risk class in W30–W90 is only about 0.56–0.63. Lowering the threshold raises recall at the cost of many false alarms (for example, in W60 reaching 80 % recall means flagging about 52 % of students).
Limitations
Windows W30–W90 and FULL are evaluated on different populations (students who unregistered before the cutoff are removed), so cross-window comparison is qualitative; FULL is not an early-prediction result.
The split is by student, not by course or year, so transfer to new courses is not evaluated.
Hyperparameters were not tuned in the reported run.
Risk thresholds are computed on the test set and are therefore optimistic; the bootstrap resamples records, not students.
Probabilities are not calibrated; a single dataset from one institution is used.
Notes on saved models

The .joblib files are pickled scikit-learn pipelines. They depend on the scikit-learn version used for training (1.9.1) and should only be loaded from trusted sources.

Author

Sofiia Velychko
