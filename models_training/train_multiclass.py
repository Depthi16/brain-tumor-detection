import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_multiclass():
    # Configuration
    IMG_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 20
    # Update this path if your data folder is different
    DATA_DIR = "c:/Users/Deepti/Desktop/brain-tumor-project/data"
    
    print("Loading Dataset...")
    # Loading dataset with categorical labels
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected Classes: {class_names}")

    # Preprocessing
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y))

    # Model Architecture
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Freeze base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax') # Softmax for multi-class
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        "c:/Users/Deepti/Desktop/brain-tumor-project/multiclass_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )
    
    print("\nTraining Complete. Model saved as multiclass_model.h5")

if __name__ == "__main__":
    train_multiclass()
