import tensorflow as tf
import os

def check_labels():
    data_dir = "c:/Users/Deepti/Desktop/brain-tumor-project/data"
    if not os.path.exists(data_dir):
        print(f"Path not found: {data_dir}")
        return
        
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(128, 128),
        batch_size=32,
        label_mode='binary'
    )
    
    print(f"Class names: {ds.class_names}")
    # Show mapping
    for i, name in enumerate(ds.class_names):
        print(f"Index {i} -> {name}")

if __name__ == "__main__":
    check_labels()
