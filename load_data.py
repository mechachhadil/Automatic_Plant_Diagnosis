import tensorflow as tf
from config import DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT, SEED

def load_datasets():
    # Dataset 1 : original
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # -----------------------------
    # Data augmentation (Dataset 2)
    # -----------------------------
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

    # -----------------------------
    # Normalisation
    # -----------------------------
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    # -----------------------------
    # Performance optimisation
    # -----------------------------
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, class_names, num_classes