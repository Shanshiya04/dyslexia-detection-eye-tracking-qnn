import numpy as np
import os
from collections import defaultdict

def build_subject_dataset(feature_file, label_file, path_file, output_name):
    X = np.load(feature_file)
    y = np.load(label_file)
    paths = np.load(path_file, allow_pickle=True)

    subject_features = defaultdict(list)
    subject_labels = {}

    for i in range(len(paths)):
        filename = os.path.basename(paths[i])
        
        parts = filename.split('_')
        subject_id = None
        for part in parts:
            if part.isdigit():
                subject_id = part
                break

        if subject_id is None:
            continue

        subject_features[subject_id].append(X[i])
        
        # Ensure label consistency
        if subject_id not in subject_labels:
            subject_labels[subject_id] = y[i]

    # 🔥 SORT SUBJECTS (VERY IMPORTANT FOR CONSISTENCY)
    subjects_sorted = sorted(subject_features.keys())

    X_subject = []
    y_subject = []

    for subject in subjects_sorted:
        avg_feature = np.mean(subject_features[subject], axis=0)
        X_subject.append(avg_feature)
        y_subject.append(subject_labels[subject])

    X_subject = np.array(X_subject)
    y_subject = np.array(y_subject)

    # ==============================
    # SAVE FILES
    # ==============================
    np.save(output_name + "_features.npy", X_subject)
    np.save(output_name + "_labels.npy", y_subject)

    # 🔥 SAVE SUBJECT IDS (CRITICAL FIX)
    np.save(output_name + "_subjects.npy", np.array(subjects_sorted))

    print(f"{output_name} subject dataset created successfully")
    print(f"Total subjects: {len(subjects_sorted)}")


# ==============================
# BUILD TRAIN + TEST
# ==============================
build_subject_dataset("train_features.npy", "train_labels.npy", "train_paths.npy", "train_subject")
build_subject_dataset("test_features.npy", "test_labels.npy", "test_paths.npy", "test_subject")