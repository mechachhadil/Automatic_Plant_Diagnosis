import os
import pickle
import tensorflow as tf
from model import build_cnn_model
from load_data import load_datasets
from config import MODEL_DIR, MODEL_FILE, HISTORY_FILE, LEARNING_RATE, EPOCHS, PATIENCE

# Charger dataset
train_dataset, val_dataset, test_dataset, class_names, num_classes = load_datasets()

# Build model
model = build_cnn_model(num_classes)

# ===============================
# PHASE 1 : Training (base gelée)
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# ===============================
# PHASE 2 : Fine-tuning
# ===============================
print("\nStarting fine-tuning...")

base_model = model.layers[0]

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# ===============================
# Merge history
# ===============================
for key in set(history.history.keys()).intersection(fine_history.history.keys()):
    history.history[key] += fine_history.history[key]
    
print(history.history.keys())
print(fine_history.history.keys())

# ===============================
# Save
# ===============================
os.makedirs(MODEL_DIR, exist_ok=True)

with open(HISTORY_FILE, "wb") as f:
    pickle.dump(history.history, f)

model.save(MODEL_FILE)

print(f"Model saved at {MODEL_FILE}")

with open("models/class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)