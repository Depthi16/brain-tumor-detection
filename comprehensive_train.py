import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import time

# --- CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
DATA_DIR = "c:/Users/Deepti/Desktop/brain-tumor-project/data_split"
RESULTS_DIR = "c:/Users/Deepti/Desktop/brain-tumor-project/training_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate():
    print("Starting Comprehensive Training Pipeline...")
    
    # 1. Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )
    
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # 2. Preprocessing
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
    test_ds_processed = test_ds.map(lambda x, y: (preprocess_input(x), y))

    # 3. Build Model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Freeze base layers
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 4. Training Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(RESULTS_DIR, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 5. Execute Training
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    # 6. Visualization: Accuracy & Loss
    h = history.history
    epochs_range = range(len(h['accuracy']))

    plt.figure(figsize=(12, 5))
    
    # Accuracy Graph
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, h['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, h['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, h['loss'], label='Training Loss')
    plt.plot(epochs_range, h['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"))
    plt.close()

    # 7. Evaluation on Test Set
    print("Evaluating on Test Set...")
    y_true = []
    y_pred_probs = []
    
    for imgs, labels in test_ds_processed:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # 8. Visualization: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > (cm.max()/2) else "black")
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    # 9. Visualization: Confidence Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_probs, bins=20, color='teal', alpha=0.7)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Probability (Tumor Presence)')
    plt.ylabel('Number of Samples')
    plt.axvline(0.5, color='red', linestyle='--')
    plt.savefig(os.path.join(RESULTS_DIR, "confidence_distribution.png"))
    plt.close()

    # 10. Save Detailed Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(RESULTS_DIR, "final_report.txt"), "w") as f:
        f.write("=== BRAIN TUMOR DETECTION REPORT ===\n")
        f.write(f"Training Time: {total_time:.2f}s\n")
        f.write(f"Final Test Accuracy: {np.mean(y_true == y_pred):.4f}\n\n")
        f.write("Classification Metrics:\n")
        f.write(report)
    
    print(f"All results and graphs saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
