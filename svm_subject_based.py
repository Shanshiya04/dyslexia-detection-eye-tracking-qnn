import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# PATHS
# ==============================
train_path = r"C:\Users\ASUS\Sample\numerical_values\train"
test_path  = r"C:\Users\ASUS\Sample\numerical_values\test"

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file_path):
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=[np.number])

    features = []
    features.extend(df.mean().values)
    features.extend(df.std().values)
    features.extend(df.min().values)
    features.extend(df.max().values)
    features.extend(df.median().values)

    return features

# ==============================
# LOAD DATA SUBJECT-WISE
# ==============================
def load_data(base_path):
    X = []
    y = []
    subject_ids = []

    class_map = {
        "control": 0,
        "dyslexic": 1
    }

    for label in class_map:
        folder = os.path.join(base_path, label)
        subject_files = defaultdict(list)

        for file in os.listdir(folder):
            if file.endswith(".csv"):
                subject_id = file.split("_")[1]
                subject_files[subject_id].append(os.path.join(folder, file))

        for subject_id, files in subject_files.items():
            subject_features = []

            for file_path in sorted(files):
                subject_features.extend(extract_features(file_path))

            X.append(subject_features)
            y.append(class_map[label])
            subject_ids.append(subject_id)

    return np.array(X), np.array(y), subject_ids

# ==============================
# LOAD DATA
# ==============================
print("Loading training data...")
X_train, y_train, train_ids = load_data(train_path)

print("Loading testing data...")
X_test, y_test, test_ids = load_data(test_path)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ==============================
# SVM PIPELINE
# ==============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=40)),
    ("svm", SVC(kernel="rbf"))
])

param_grid = {
    "svm__C": [1, 10, 50, 100],
    "svm__gamma": ["scale", 0.1, 0.01, 0.001]
}

print("\nTraining SVM...")
grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

# ==============================
# TESTING
# ==============================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nFinal Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==============================
# SAVE SUBJECT PREDICTIONS
# ==============================
class_names = ["control", "dyslexic"]
pred_labels = [class_names[p] for p in y_pred]

df = pd.DataFrame({
    "subject_id": test_ids,
    "prediction": pred_labels
})

df.to_csv("svm_subject_predictions.csv", index=False)
print("\n✅ svm_subject_predictions.csv saved!")
print(df)