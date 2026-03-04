import os
import numpy as np
import cv2
from glob import glob

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

import config
import pickle

# ===============================
# LOAD DATASET
# ===============================

IMAGE_SIZE = 64
DATASET_PATH = config.DATASET_PATH

X = []
y = []

class_folders = sorted(os.listdir(DATASET_PATH))

for label, folder in enumerate(class_folders):
    folder_path = os.path.join(DATASET_PATH, folder)
    images = glob(os.path.join(folder_path, "*.jpg"))

    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img.flatten())
        y.append(label)

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)

print("Dataset shape:", X.shape)
print("Number of classes:", len(class_folders))

# ===============================
# TRAIN / TEST SPLIT (Stratified)
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=config.SEED
)

# ===============================
# PIPELINE (Scaler + PCA + SVM)
# ===============================

ml_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=150, random_state=config.SEED)),
    ("svm", SVC(kernel="rbf", probability=True, random_state=config.SEED))
])

# ===============================
# TRAIN
# ===============================

ml_pipeline.fit(X_train, y_train)

# ===============================
# TEST EVALUATION
# ===============================

y_pred = ml_pipeline.predict(X_test)
y_scores = ml_pipeline.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("\n=== Test Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Weighted: {f1_weighted:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_folders))

# ===============================
# ROC-AUC (Multiclass OvR)
# ===============================

y_test_bin = label_binarize(y_test, classes=range(len(class_folders)))

try:
    auc_score = roc_auc_score(y_test_bin, y_scores, average="macro", multi_class="ovr")
    print(f"AUC Macro (OvR): {auc_score:.4f}")
except ValueError:
    print("AUC could not be computed.")

# ===============================
# STRATIFIED K-FOLD CROSS VALIDATION
# ===============================

print("\nRunning 5-Fold Stratified Cross Validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

cv_scores = cross_val_score(
    ml_pipeline,
    X,
    y,
    cv=skf,
    scoring="accuracy",
    n_jobs=1
)

print(f"K-Fold Accuracy Mean: {cv_scores.mean():.4f}")
print(f"K-Fold Accuracy Std: {cv_scores.std():.4f}")

print("\nML pipeline evaluation complete.")

with open("models/ml_accuracy.pkl", "wb") as f:
    pickle.dump(accuracy, f)