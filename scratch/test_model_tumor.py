import tensorflow as tf
import os
import numpy as np
import cv2

def test_model():
    model_path = "c:/Users/Deepti/Desktop/brain-tumor-project/mobilenet_model.h5"
    if not os.path.exists(model_path):
        print("Model not found")
        return
        
    model = tf.keras.models.load_model(model_path)
    data_dir = "c:/Users/Deepti/Desktop/brain-tumor-project/data/tumor"
    
    files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:5]
    print(f"Testing on {len(files)} tumor images...")
    
    for f in files:
        img_path = os.path.join(data_dir, f)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_resized))
        img_final = np.expand_dims(img_preprocessed, axis=0)
        
        pred = model.predict(img_final, verbose=0)[0][0]
        print(f"File: {f}, Prediction: {pred:.4f}")

if __name__ == "__main__":
    test_model()
