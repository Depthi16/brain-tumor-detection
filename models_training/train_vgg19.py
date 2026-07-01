import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def train():
    IMG_SIZE = 224
    BATCH_SIZE = 16
    train_ds = tf.keras.utils.image_dataset_from_directory("../data", validation_split=0.2, subset="training", seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='binary')
    val_ds = tf.keras.utils.image_dataset_from_directory("../data", validation_split=0.2, subset="validation", seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='binary')
    preprocess = tf.keras.applications.vgg19.preprocess_input
    train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y))
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
    model = Sequential([base_model, Flatten(), Dense(4096, activation='relu'), Dropout(0.5), Dense(4096, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid')])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint("../vgg19_model.h5", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[checkpoint])

if __name__ == "__main__":
    train()
