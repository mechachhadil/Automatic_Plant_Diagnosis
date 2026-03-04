import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from load_data import load_datasets
from config import MODEL_FILE

# Load model
model = tf.keras.models.load_model(MODEL_FILE)

# Load dataset
_, val_dataset, class_names, _ = load_datasets()

y_true, y_pred = [], []

for images, labels in val_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))