import tensorflow as tf
import os
import numpy as np
import cv2
import time
import sys

def benchmark():
    # Allow model selection via command line argument, otherwise check in order
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_paths = [
            "c:/Users/Deepti/Desktop/brain-tumor-project/efficientnet_model.h5",
            "c:/Users/Deepti/Desktop/brain-tumor-project/mobilenet_model.h5",
            "c:/Users/Deepti/Desktop/brain-tumor-project/resnet50_model.h5"
        ]
        model_path = next((path for path in model_paths if os.path.exists(path)), None)

    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model not found at path: {model_path}")
        print("Please place a valid model file or specify the path as an argument.")
        return
        
    print(f"Loading model: {model_path}")
    start_load = time.time()
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded in {time.time() - start_load:.4f} seconds")
    
    # Determine model type and parameters dynamically
    filename = os.path.basename(model_path).lower()
    if "efficientnet" in filename:
        img_size = (224, 224)
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
        model_name = "EfficientNetB0"
    elif "resnet50" in filename:
        img_size = (224, 224)
        preprocess_fn = lambda x: tf.keras.applications.resnet50.preprocess_input(x.astype('float32'))
        model_name = "ResNet50"
    elif "mobilenet" in filename or "best_model" in filename:
        img_size = (128, 128)
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
        model_name = "MobileNetV2"
    else:
        # Default fallback
        img_size = (224, 224)
        preprocess_fn = lambda x: x / 255.0
        model_name = "Unknown Model"

    print(f"Detected model type: {model_name} (Expected input shape: {img_size})")

    # Create a dummy image matching the model's expected shape
    img = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    img_preprocessed = preprocess_fn(np.array(img_resized))
    img_final = np.expand_dims(img_preprocessed, axis=0)
    
    # Warmup
    print("Warming up model.predict...")
    model.predict(img_final, verbose=0)
    print("Warming up model call...")
    model(img_final, training=False)
    
    # Benchmark predict
    runs = 10
    start = time.time()
    for _ in range(runs):
        model.predict(img_final, verbose=0)
    predict_time = (time.time() - start) / runs
    print(f"Average time for model.predict: {predict_time:.4f} seconds")
    
    # Benchmark direct call
    start = time.time()
    for _ in range(runs):
        model(img_final, training=False).numpy()
    direct_time = (time.time() - start) / runs
    print(f"Average time for direct call: {direct_time:.4f} seconds")
    
    print(f"Speedup: {predict_time / direct_time:.2f}x")

if __name__ == "__main__":
    benchmark()

