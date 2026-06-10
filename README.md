# Spam Mail Prediction

A machine learning project that classifies emails as **spam** or **ham (not spam)** using Logistic Regression and TF-IDF feature extraction.

---

## Overview

This project trains a binary text classifier on a labeled email dataset. Given any email message, the model predicts whether it is spam or legitimate mail with ~96.7% accuracy.

---

## Dataset

| Property | Value |
|---|---|
| File | `mail_data.csv` |
| Total samples | 5,572 |
| Ham (legitimate) | 4,825 |
| Spam | 747 |
| Columns | `Category`, `Message` |

Labels: `spam → 0`, `ham → 1`

---

## Project Structure

```
├── mail_data.csv              # Labeled email dataset
├── model.py                   # Training script + model export (.joblib)
├── spam_mail_prediction.py    # Training + inline prediction example
├── requirements.txt           # Python dependencies
└── README.md
```

---

## How It Works

1. **Load data** — Read `mail_data.csv` and encode labels (`spam=0`, `ham=1`).
2. **Split** — 80% training / 20% test (`random_state=3`).
3. **Feature extraction** — Convert email text to numerical vectors using `TfidfVectorizer` (removes English stop words, lowercases text).
4. **Train** — Fit a `LogisticRegression` model on the TF-IDF features.
5. **Evaluate** — Measure accuracy on both train and test sets.
6. **Predict** — Transform new email text and run `model.predict()`.

---

## Model Performance

| Split | Accuracy |
|---|---|
| Training set | 96.77% |
| Test set | 96.68% |

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

---

## Usage

### Train and export the model

```bash
python model.py
```

This trains the model and saves it to `model.joblib` for reuse.

### Run a prediction inline

```bash
python spam_mail_prediction.py
```

To classify a custom email, edit the `input_mail` list at the bottom of `spam_mail_prediction.py`:

```python
input_mail = ["Your email text goes here..."]
input_features = feature_extraction.transform(input_mail)
output = model.predict(input_features)

if output == 0:
    print('Spam Mail')
else:
    print('Not Spam')
```

### Load the saved model (after running `model.py`)

```python
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load('model.joblib')
# Note: re-fit the vectorizer or save/load it separately for standalone inference
```

---

## Notes

- The `TfidfVectorizer` must be fit on training data and reused for any new predictions — it is not saved separately in the current scripts. For production use, save the vectorizer alongside the model:
  ```python
  joblib.dump(feature_extraction, 'vectorizer.joblib')
  ```
- The dataset is imbalanced (~87% ham, ~13% spam). If false negatives are a concern, consider threshold tuning or class weighting (`LogisticRegression(class_weight='balanced')`).
