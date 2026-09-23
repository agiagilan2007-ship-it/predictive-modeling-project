[README.md](https://github.com/user-attachments/files/32562803/README.md)
# Predictive Modeling Using Machine Learning

A supervised learning project that builds and compares three classification
models — **Logistic Regression**, **Decision Tree**, and **Random Forest** —
to predict whether a tumor is malignant or benign, using the Breast Cancer
Wisconsin (Diagnostic) dataset. The project covers the full workflow: data
loading, train/test splitting, model training, and evaluation with accuracy
scores, classification reports, confusion matrices, and ROC curves.

## 📌 Project Overview

- **Task**: Binary classification (predict tumor diagnosis)
- **Dataset**: `sklearn.datasets.load_breast_cancer` — 569 samples, 30 numeric
  features (mean, standard error, and "worst" measurements of cell nuclei),
  built into scikit-learn so no external download is needed
- **Algorithms compared**: Logistic Regression, Decision Tree, Random Forest
- **Evaluation**: Accuracy, precision/recall/F1, confusion matrix, ROC curve
  and AUC

## 🗂️ Project Structure

```
predictive-modeling-project/
├── src/
│   └── train_model.py       # Loads data, trains models, evaluates, plots
├── outputs/                 # Generated after running the script
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── model_comparison.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/predictive-modeling-ml.git
cd predictive-modeling-ml
pip install -r requirements.txt
```

## ▶️ Usage

Run the training script from the project root:

```bash
python src/train_model.py
```

This will:
1. Load the dataset and split it 80/20 into train and test sets (stratified)
2. Scale features for Logistic Regression (tree-based models don't need it)
3. Train Logistic Regression, Decision Tree, and Random Forest
4. Print accuracy, ROC AUC, and a full classification report for each model
5. Save two comparison plots and a results CSV to `outputs/`

## 📊 Results

| Model               | Accuracy | ROC AUC |
|---------------------|----------|---------|
| Logistic Regression | ~0.98    | ~0.995  |
| Random Forest       | ~0.96    | ~0.993  |
| Decision Tree       | ~0.92    | ~0.916  |

*(Exact numbers may vary slightly by environment/library version; the script
also writes these to `outputs/model_comparison.csv`.)*

**Confusion matrices** for all three models, side by side:

`outputs/confusion_matrices.png`

**ROC curves** for all three models on one plot, for direct comparison:

`outputs/roc_curves.png`

## 🧠 What This Demonstrates

- Splitting data into training and test sets, and why scaling matters for
  some algorithms (Logistic Regression) but not others (tree-based models)
- Training multiple classical ML algorithms on the same problem
- Comparing models using more than one metric (accuracy alone can be
  misleading, especially on imbalanced data)
- Reading and interpreting confusion matrices and ROC/AUC curves

## 🔧 Possible Extensions

- Add cross-validation (`GridSearchCV`) for hyperparameter tuning
- Add feature importance plots for the Random Forest
- Try additional models (SVM, Gradient Boosting, XGBoost)
- Swap in a different dataset (e.g. customer churn, loan default) to
  practice the same pipeline on new data

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
