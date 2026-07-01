import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def train():
    # --- SMART PATH DETECTION ---
    if os.path.exists("data"):
        DATA_PATH = "data"
        MODEL_FILE = "mobilenet_model.h5"
    elif os.path.exists("../data"):
        DATA_PATH = "../data"
        MODEL_FILE = "../mobilenet_model.h5"
    else:
        print("❌ Error: Could not find 'data' folder. Please make sure you are in the project directory.")
        return

    print(f"🚀 Preparing MobileNetV2 training pipeline (Data: {DATA_PATH})...")

    # --- CONFIGURATION ---
    IMG_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 35

    # 1. Load Datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # 2. Preprocessing & Augmentation
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    augmentation = Sequential([
        RandomFlip("horizontal"),  # Clinically accurate horizontal flip only
        RandomRotation(0.1),
        RandomZoom(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 3. Load existing model OR Build new one
    if os.path.exists(MODEL_FILE):
        print(f"♻️  Existing model found at {MODEL_FILE}. Resuming training...")
        model = load_model(MODEL_FILE)
        print("🔧 Recompiling model to synchronize optimizer...")
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        print("🆕 No existing model found. Building new MobileNetV2 model...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        
        # Keep base frozen for stable feature extraction
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # 4. Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_FILE, 
        monitor="val_accuracy", 
        save_best_only=True, 
        mode="max", 
        verbose=1
    )
    
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

    # 5. Execute Training
    print("✨ Training in progress... Aiming for stable accuracy!")
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=[checkpoint, lr_reducer, early_stop]
    )

    print(f"✅ Training session complete! Best model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
