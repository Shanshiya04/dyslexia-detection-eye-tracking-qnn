import pandas as pd
from collections import Counter

print("\n🔷 MAJORITY VOTING (NEGLECT TIES) 🔷")

# Load merged file (your test_result.csv)
df = pd.read_csv("test_result.csv")

final_results = []

for i, row in df.iterrows():
    votes = [
        row["SVM"],
        row["Logistic Regression"],
        row["ResNet18"],
        row["QNN"]
    ]
    
    count = Counter(votes)
    most_common = count.most_common()
    
    # Check tie
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        final_results.append("tie")
    else:
        final_results.append(most_common[0][0])

df["Final Result"] = final_results

# Save
df.to_csv("final_majority_neg.csv", index=False)

print("\nFinal Results (Tie = Neglected):")
print(df)

# Accuracy (excluding ties)
valid_df = df[df["Final Result"] != "tie"]

if len(valid_df) > 0:
    acc = sum(valid_df["Final Result"] == valid_df["Actual value"]) / len(valid_df) * 100
    print(f"\nAccuracy (excluding ties): {acc:.2f}%")
else:
    print("\nNo valid predictions (all ties)")