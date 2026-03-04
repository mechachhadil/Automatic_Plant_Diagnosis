import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import config

# ===============================
# Reload validation dataset
# ===============================
raw_val_dataset = tf.keras.utils.image_dataset_from_directory(
    config.DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=config.SEED,
    image_size=(128, 128),
    batch_size=32
)

# Save class names BEFORE normalization
class_names = raw_val_dataset.class_names

# Normalisation
normalization_layer = tf.keras.layers.Rescaling(1./255)
val_dataset = raw_val_dataset.map(lambda x, y: (normalization_layer(x), y))

val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# ===============================
# Paths & Directories
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_cnn.h5")
HISTORY_PATH = os.path.join(BASE_DIR, "models", "training_history.pkl")
ML_ACC_PATH = os.path.join(BASE_DIR, "models", "ml_accuracy.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ===============================
# Load training history
# ===============================
with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

# ===============================
# Plot Accuracy
# ===============================
plt.figure()
plt.plot(history["accuracy"], marker='o')
plt.plot(history["val_accuracy"], marker='o')
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.grid(True)
acc_path = os.path.join(FIGURES_DIR, "cnn_accuracy.png")
plt.savefig(acc_path)
plt.show()

# ===============================
# Plot Loss
# ===============================
plt.figure()
plt.plot(history["loss"], marker='o')
plt.plot(history["val_loss"], marker='o')
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.grid(True)
loss_path = os.path.join(FIGURES_DIR, "cnn_loss.png")
plt.savefig(loss_path)
plt.show()

# ===============================
# Load CNN model
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)

# ===============================
# Confusion Matrix on Validation Set (Optimized)
# ===============================

print("Running full validation prediction...")

# Get true labels
y_true = np.concatenate([y for x, y in val_dataset], axis=0)

# Predict all batches at once
y_probs = model.predict(val_dataset, verbose=1)
y_pred = np.argmax(y_probs, axis=1)

#cm = confusion_matrix(y_true, y_pred)
#plt.figure(figsize=(12,10))
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#disp.plot(cmap=plt.cm.Blues)
#plt.title("CNN Confusion Matrix")
#cm_path = os.path.join(FIGURES_DIR, "cnn_confusion_matrix.png")
#plt.savefig(cm_path)
#plt.show()

# ===============================
# Classification metrics per class
# ===============================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

recalls = []
f1_scores = []

for class_name in class_names:
    recalls.append(report[class_name]["recall"])
    f1_scores.append(report[class_name]["f1-score"])

recalls = np.array(recalls)
f1_scores = np.array(f1_scores)

# Get indices of 10 worst recall classes
worst_indices = np.argsort(recalls)[:10]

worst_classes = [class_names[i] for i in worst_indices]
worst_recalls = recalls[worst_indices]

print("\nTop-10 Worst Recall Classes:")
for c, r in zip(worst_classes, worst_recalls):
    print(f"{c} → Recall: {r:.3f}")

# ===============================
# Plot Worst Recall Classes
# ===============================

plt.figure(figsize=(10,6))
plt.barh(worst_classes, worst_recalls)
plt.xlabel("Recall")
plt.title("Top-10 Classes with Worst Recall")
plt.xlim(0, 1)

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(os.path.join(FIGURES_DIR, "worst_recall_top10.png"), dpi=300)
plt.show()

# ===============================
# F1-score per class
# ===============================

plt.figure(figsize=(14,6))
plt.bar(range(len(class_names)), f1_scores)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90,
    fontsize=6
)

plt.ylabel("F1-score")
plt.title("F1-score per Class")
plt.ylim(0,1)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "f1_per_class.png"), dpi=300)
plt.show()

# ===============================
# Load ML (PCA+SVM) accuracy
# ===============================
if os.path.exists(ML_ACC_PATH):
    with open(ML_ACC_PATH, "rb") as f:
        svm_acc = pickle.load(f)
else:
    print("ml_accuracy.pkl not found. Using default value 0.65")
    svm_acc = 0.65

# ===============================
# Bar Chart Comparison
# ===============================
cnn_acc = history["val_accuracy"][-1]  # Last accuracy validation CNN

plt.figure()
plt.bar(["CNN", "PCA+SVM"], [cnn_acc, svm_acc], color=["blue", "orange"])
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate([cnn_acc, svm_acc]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
bar_path = os.path.join(FIGURES_DIR, "model_comparison.png")
plt.savefig(bar_path)
plt.show()

print(f"\nFigures saved in {FIGURES_DIR}")
print("Done plotting results.")