import os

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "plantvillage_dataset", "color")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "plant_cnn.keras")
HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.pkl")

# ==============================
# Dataset parameters
# ==============================
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
SEED = 42

# ==============================
# CNN parameters
# ==============================
NUM_CLASSES = None  # à initialiser après chargement dataset
INPUT_SHAPE = (128, 128, 3)
LEARNING_RATE = 0.0005
EPOCHS = 30
PATIENCE = 5  # early stopping