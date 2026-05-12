import pandas as pd
from collections import Counter

# LOAD PREDICTIONS
cnn = pd.read_csv("cnn_subject_predictions.csv")
log = pd.read_csv("logistic_subject_predictions.csv")
svm = pd.read_csv("svm_subject_predictions.csv")

cnn["subject_id"] = cnn["subject_id"].astype(str)
log["subject_id"] = log["subject_id"].astype(str)
svm["subject_id"] = svm["subject_id"].astype(str)

# Use CNN test subjects as reference
test_subjects = cnn["subject_id"].tolist()

log = log[log["subject_id"].isin(test_subjects)]
svm = svm[svm["subject_id"].isin(test_subjects)]

cnn = cnn.rename(columns={"prediction": "cnn"})
log = log.rename(columns={"prediction": "logistic"})
svm = svm.rename(columns={"prediction": "svm"})

df = cnn.merge(log, on=["subject_id", "true_label"]).merge(svm, on=["subject_id", "true_label"])

print("\n🧠 MULTIMODAL FUSION RESULTS")
print("=================================")

correct = 0
total = 0
rows = []

for _, row in df.iterrows():
    votes = [row["cnn"], row["logistic"], row["svm"]]
    final_pred = Counter(votes).most_common(1)[0][0]

    print(f"Subject {row['subject_id']} → True: {row['true_label']}, Final Predicted: {final_pred}")

    rows.append({
        "subject_id": row["subject_id"],
        "true_label": row["true_label"],
        "prediction": final_pred
    })

    if final_pred == row["true_label"]:
        correct += 1
    total += 1

if total > 0:
    accuracy = 100 * correct / total
    print(f"\n🔥 Multimodal Subject-level Accuracy: {accuracy:.2f}%")

    # SAVE ACCURACY FOR MASTER SCRIPT
    with open("fusion_accuracy.txt", "w") as f:
        f.write(f"{accuracy:.2f}")

    # SAVE FUSION PREDICTIONS CSV (OPTIONAL)
    pd.DataFrame(rows).to_csv("fusion_subject_predictions.csv", index=False)

else:
    print("\n⚠️ No overlapping subjects found.")
