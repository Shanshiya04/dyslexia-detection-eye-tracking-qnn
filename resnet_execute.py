import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict, Counter
import pandas as pd
import os

# ==============================
# PATHS
# ==============================
data_dir = r"C:\Users\ASUS\Sample\eye_dataset"
model_path = "resnet18_dyslexia_finetuned.pth"

# ==============================
# TRANSFORMS
# ==============================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(data_dir + "/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class_names = test_dataset.classes

# ==============================
# LOAD MODEL
# ==============================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

print("Model loaded successfully!")

# ==============================
# IMAGE-LEVEL EVALUATION
# ==============================
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.append(labels.item())
        y_pred.append(predicted.item())

print("\nIMAGE LEVEL RESULTS")
print("====================")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=class_names))

# ==============================
# SUBJECT-LEVEL PREDICTION
# ==============================
subject_preds = defaultdict(list)
subject_true = {}

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        img_path = test_dataset.samples[i][0]
        filename = os.path.basename(img_path)

        subject_id = filename.replace("Subject_", "").split("_")[0]

        pred_label = class_names[predicted.item()]
        true_label = class_names[labels.item()]

        subject_preds[subject_id].append(pred_label)
        subject_true[subject_id] = true_label

rows = []
correct = 0
total = 0

print("\nSUBJECT LEVEL RESULTS")
print("====================")

for subject_id in sorted(subject_preds.keys()):
    final_pred = Counter(subject_preds[subject_id]).most_common(1)[0][0]
    true_label = subject_true[subject_id]

    print(f"Subject {subject_id} → True: {true_label}, Predicted: {final_pred}")

    rows.append({
        "subject_id": subject_id,
        "prediction": final_pred
    })

    if final_pred == true_label:
        correct += 1
    total += 1

subject_accuracy = 100 * correct / total
print(f"\nSubject-level Accuracy: {subject_accuracy:.2f}%")

pd.DataFrame(rows).to_csv("cnn_subject_predictions.csv", index=False)
print("Saved: cnn_subject_predictions.csv")