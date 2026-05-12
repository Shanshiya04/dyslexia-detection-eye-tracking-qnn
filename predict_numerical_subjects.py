import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ===============================
# LOAD DATASET
# ===============================

df = pd.read_csv("numerical_subject_dataset.csv")

if df.empty:
    print("❌ Dataset is empty. Run create_numerical_dataset.py first.")
    exit()

df["subject_id"] = df["subject_id"].astype(str).str.strip()

print(f"📊 Total subjects in dataset: {len(df)}")

# ===============================
# LOAD COMMON SPLIT
# ===============================

train_ids = pd.read_csv("split/train_subjects.csv")["subject_id"].astype(str).str.strip()
test_ids = pd.read_csv("split/test_subjects.csv")["subject_id"].astype(str).str.strip()

print("\nDEBUG:")
print("Train subjects count:", len(train_ids))
print("Test subjects count:", len(test_ids))

# ===============================
# SPLIT USING SUBJECT IDS
# ===============================

train_df = df[df["subject_id"].isin(train_ids)]
test_df = df[df["subject_id"].isin(test_ids)]

print("Numerical train subjects:", len(train_df))
print("Numerical test subjects:", len(test_df))

if len(train_df) == 0 or len(test_df) == 0:
    print("❌ Train/Test split failed. Check dataset creation.")
    exit()

# ===============================
# FEATURES & LABELS
# ===============================

X_train = train_df.drop(columns=["subject_id", "label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["subject_id", "label"])
y_test = test_df["label"]
id_test = test_df["subject_id"]

# ===============================
# SCALING
# ===============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# LOGISTIC REGRESSION
# ===============================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# ===============================
# SVM
# ===============================

svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# ===============================
# LABEL MAP
# ===============================

label_map = {0: "control", 1: "dyslexic"}

# ===============================
# SAVE RESULTS
# ===============================

log_rows = []
svm_rows = []

print("\n🧠 NUMERICAL SUBJECT RESULTS (LOGISTIC)")
print("=================================")

for sid, true, pred in zip(id_test, y_test, log_pred):
    true_label = label_map[true]
    pred_label = label_map[pred]

    print(f"Subject {sid} → True: {true_label}, Predicted: {pred_label}")

    log_rows.append({
        "subject_id": sid,
        "true_label": true_label,
        "prediction": pred_label
    })

print("\n🧠 NUMERICAL SUBJECT RESULTS (SVM)")
print("=================================")

for sid, true, pred in zip(id_test, y_test, svm_pred):
    true_label = label_map[true]
    pred_label = label_map[pred]

    print(f"Subject {sid} → True: {true_label}, Predicted: {pred_label}")

    svm_rows.append({
        "subject_id": sid,
        "true_label": true_label,
        "prediction": pred_label
    })

# ===============================
# SAVE CSV
# ===============================

pd.DataFrame(log_rows).to_csv("logistic_subject_predictions.csv", index=False)
pd.DataFrame(svm_rows).to_csv("svm_subject_predictions.csv", index=False)

print("\nSaved: logistic_subject_predictions.csv")
print("Saved: svm_subject_predictions.csv")

# ===============================
# ACCURACY
# ===============================

log_correct = sum(1 for r in log_rows if r["true_label"] == r["prediction"])
svm_correct = sum(1 for r in svm_rows if r["true_label"] == r["prediction"])

total = len(log_rows)

log_acc = 100 * log_correct / total
svm_acc = 100 * svm_correct / total

print(f"\nLogistic Accuracy: {log_acc:.2f}%")
print(f"SVM Accuracy: {svm_acc:.2f}%")

with open("logistic_accuracy.txt", "w") as f:
    f.write(f"{log_acc:.2f}")

with open("svm_accuracy.txt", "w") as f:
    f.write(f"{svm_acc:.2f}")