## Breast Cancer Classification Demo

This small Python project trains and evaluates three classification models on the built-in scikit-learn breast cancer dataset.

What the project does
- Loads the `load_breast_cancer` dataset from scikit-learn.
- Splits the data into train and test sets (80/20, `random_state=42`).
- Scales features for Logistic Regression using `StandardScaler`.
- Trains three models:
  - Logistic Regression (with feature scaling)
  - Decision Tree
  - Random Forest
- Prints evaluation metrics for each model: Accuracy, Precision, Recall, F1 score, and the Confusion Matrix.

Files
- `main.py` — The script that loads data, trains the models, and prints evaluation metrics.
- `requirements.txt` — Minimal Python packages required to run the script.

Requirements
- Python 3.8+ (tested with 3.8/3.9/3.10)
- See `requirements.txt` for the exact packages. Installing from that file will pull in scikit-learn and its dependencies (including NumPy).

Install
Open PowerShell and run:

```powershell
python -m pip install --upgrade pip; \
python -m pip install -r requirements.txt
```

Run
From the project root (where `main.py` is located) run:

```powershell
python main.py
```

What to expect
- The script trains three models and prints performance metrics to the console. The Logistic Regression model uses feature scaling; the tree-based models are trained on the raw features in `main.py`.
- Example output (truncated) — you'll see three blocks, one per model, similar to:

```
Logistic Regression Performance:
Accuracy: 0.9649
Precision: 0.9714
Recall: 0.9623
F1 Score: 0.9668
Confusion Matrix:
[[39  1]
 [ 3 71]]
```

Notes and small improvements
- You can optionally scale features before training tree-based models, but tree models typically don't require scaling.
- To reproduce exact results, the script already fixes `random_state=42` for train/test split and for tree/forest models.

License & Credits
- This project is a small demo using scikit-learn. Use and modify as you like.

Contact
- If you need help extending this demo (adding cross-validation, hyperparameter tuning, or saving trained models), open an issue or ask for help.
