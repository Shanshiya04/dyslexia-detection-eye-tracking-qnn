import torch
import numpy as np
import pandas as pd
from qnn_circuit import QNNModel
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# LOAD SUBJECT LEVEL DATA
# ==============================
X = np.load("test_subject_features.npy")
y = np.load("test_subject_labels.npy")

# ==============================
# LOAD SUBJECT IDS (SAFE METHOD)
# ==============================
# IMPORTANT: ensure this file is created in build_subject_dataset.py
subject_ids = np.load("test_subject_subjects.npy", allow_pickle=True)
subject_ids = subject_ids.astype(str)

# ==============================
# LOAD PCA
# ==============================
pca = joblib.load("pca_model.pkl")
X = pca.transform(X)
X = torch.tensor(X, dtype=torch.float32)

# ==============================
# LOAD MODEL
# ==============================
model = QNNModel()
model.load_state_dict(torch.load("qnn_model.pth"))
model.eval()

outputs = model(X)
_, preds = torch.max(outputs, 1)

print("\nQNN SUBJECT LEVEL RESULTS")
print("==========================")

class_names = ["control", "dyslexic"]

correct = 0
results = []

for i in range(len(y)):
    subject = subject_ids[i]
    true_label = class_names[y[i]]
    pred_label = class_names[preds[i]]

    print(f"Subject {subject} → True: {true_label}, Predicted: {pred_label}")

    results.append([subject, pred_label])

    if preds[i] == y[i]:
        correct += 1

accuracy = 100 * correct / len(y)
print(f"\nSubject-level Accuracy: {accuracy:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y, preds))

print("\nClassification Report:")
print(classification_report(y, preds, target_names=class_names))

# ==============================
# SAVE CSV
# ==============================
df = pd.DataFrame(results, columns=["subject_id", "prediction"])
df.to_csv("qnn_subject_predictions.csv", index=False)

print("\n✅ qnn_subject_predictions.csv created successfully!")