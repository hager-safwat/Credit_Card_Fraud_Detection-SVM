# Credit Card Fraud Detection (SVM)

A machine-learning project on detecting fraudulent credit-card transactions
with a Support Vector Machine, on a strongly imbalanced dataset (fraud is
about 0.17 % of all transactions).

## What the notebook covers

- **Exploratory analysis:** class balance, transaction amounts and times per
  class, correlations, outliers on V1 and V2.
- **Visualisation:** PCA and t-SNE projections coloured by class.
- **Modelling:** a stratified train / test split, scaling fitted on the
  training set only, and an RBF SVM with balanced class weights tuned by
  5-fold grid search on C and gamma, scored with average precision.
- **Evaluation:** precision, recall, F1, confusion matrix and the area under
  the precision-recall curve on the untouched, fully imbalanced test set
  (accuracy alone is misleading when 99.8 % of the samples are normal).

To keep the SVM tractable (its cost grows roughly quadratically with the
number of samples), it is fitted on all frauds plus 5000 randomly sampled
normal transactions from the training set. The test set is not resampled.

## Status

The exploratory and visualisation cells contain the outputs of the original
run. The modelling and evaluation cells were completed afterwards and their
code has been checked end to end on a synthetic table with the same columns,
but they have **not yet been run on the real dataset**, so no results are
reported here. Running the notebook with the real file fills them in.

Note: the original run used an incomplete copy of the CSV (219,902 of the
284,807 rows, with one row containing missing values). The modelling cell
drops incomplete rows.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Machine Learning Group, Université Libre de Bruxelles): 284,807
transactions by European cardholders over two days in September 2013, 492 of
them fraudulent. Features `V1` to `V28` are PCA components; `Time`, `Amount`
and the label `Class` are original. Download `creditcard.csv` from Kaggle
(free account needed) and place it next to the notebook. It is not included
in this repository.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

The t-SNE cell is slow on the full dataset (it can take a long time on a
laptop); it can be skipped without affecting the model.

## Files

- `Credit_Card_Fraud_Detection.ipynb`: analysis, model and evaluation.
- `requirements.txt`: Python packages.
