import pandas as pd
from collections import Counter

print("\n🔷 MAJORITY VOTING (ACCURACY DECISION CONTROL) 🔷")

df = pd.read_csv("test_result.csv")

final_results = []

for i, row in df.iterrows():
    votes = [
        ("SVM", row["SVM"]),
        ("Logistic Regression", row["Logistic Regression"]),
        ("ResNet18", row["ResNet18"]),
        ("QNN", row["QNN"])
    ]
    
    vote_values = [v[1] for v in votes]
    count = Counter(vote_values)
    most_common = count.most_common()
    
    # If no tie → normal majority
    if not (len(most_common) > 1 and most_common[0][1] == most_common[1][1]):
        final_results.append(most_common[0][0])
    
    else:
        # Tie → choose highest accuracy model
        priority = ["SVM", "Logistic Regression", "ResNet18", "QNN"]
        
        chosen = None
        for model in priority:
            for m, val in votes:
                if m == model:
                    chosen = val
                    break
            if chosen:
                break
        
        final_results.append(chosen)

df["Final Result"] = final_results

# Save
df.to_csv("final_majority_adc.csv", index=False)

print("\nFinal Results (Tie resolved by accuracy):")
print(df)

# Accuracy
acc = sum(df["Final Result"] == df["Actual value"]) / len(df) * 100
print(f"\nFinal Fusion Accuracy: {acc:.2f}%")