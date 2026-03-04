# train.py
import os
import pickle
import tensorflow as tf
from model import build_cnn_model
from load_data import load_datasets
from config import MODEL_DIR, MODEL_FILE, HISTORY_FILE, LEARNING_RATE, EPOCHS, PATIENCE

# Charger dataset
train_dataset, val_dataset, class_names, num_classes = load_datasets()

# Build model
model = build_cnn_model(num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Callbacks avancés
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
os.makedirs(MODEL_DIR, exist_ok=True)

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# Sauvegarder history
with open(HISTORY_FILE, "wb") as f:
    pickle.dump(history.history, f)

# Sauvegarder modèle
model.save(MODEL_FILE)
print(f"Model saved at {MODEL_FILE}")