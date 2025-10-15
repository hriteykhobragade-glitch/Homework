from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standard data division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)

# Model 2: Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

# Model 3: Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:", flush=True)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}", flush=True)
    print(f"Precision: {precision_score(y_true, y_pred):.4f}", flush=True)
    print(f"Recall: {recall_score(y_true, y_pred):.4f}", flush=True)
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}", flush=True)
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}", flush=True)

# Evaluate all models
evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Decision Tree", y_test, tree_preds)
evaluate_model("Random Forest", y_test, rf_preds)