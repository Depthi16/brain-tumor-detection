import os
import cv2
import numpy as np
from flask import Flask, render_template, request, url_for, session, redirect, flash, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import mysql.connector

app = Flask(__name__)
app.secret_key = "secret123"

# Create uploads folder if not exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# DATABASE SETUP
def get_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="brain_tumor_db"
    )
    return conn


# Load Models
models = {}
def load_app_models():
    try:
        if os.path.exists("multiclass_model.h5"):
            models['multiclass'] = load_model("multiclass_model.h5")
            print("Multi-class model loaded.")
        
        # Binary models - prioritize EfficientNetB0
        if os.path.exists("efficientnet_model.h5"):
            models['efficientnet'] = load_model("efficientnet_model.h5")
            print("EfficientNetB0 loaded.")
        elif os.path.exists("mobilenet_model.h5"):
            models['mobilenet'] = load_model("mobilenet_model.h5")
            print("MobileNetV2 loaded.")
        elif os.path.exists("resnet50_model.h5"):
            try:
                models['resnet50'] = load_model("resnet50_model.h5")
                print("ResNet50 loaded.")
            except Exception as e:
                print(f"ResNet50 too large for memory, skipping: {e}")
    except Exception as e:
        print(f"Error loading models: {e}")

load_app_models()

def is_brain_mri(img):
    """Simple heuristics to determine if image is likely a brain MRI"""
    # 1. MRIs are grayscale. Even if saved as RGB, channels should be identical.
    b, g, r = cv2.split(img)
    color_diff = np.mean(np.abs(r.astype(int) - g.astype(int))) + np.mean(np.abs(r.astype(int) - b.astype(int)))
    if color_diff > 5.0:
        return False
        
    # 2. MRI scans typically have dark backgrounds. Check corners.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corner_brightness = np.mean([
        np.mean(gray[0:10, 0:10]),
        np.mean(gray[0:10, w-10:w]),
        np.mean(gray[h-10:h, 0:10]),
        np.mean(gray[h-10:h, w-10:w])
    ])
    if corner_brightness > 75:
        return False
        
    return True

def draw_highlight(filepath, filename):
    """Helper to draw tumor bounding box/contour"""
    try:
        orig_img = cv2.imread(filepath)
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Dynamic threshold based on image brightness
        threshold_value = max(100, np.max(blur) * 0.7)
        _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            tumor_color = (100, 150, 255) # Light blue/orange
            overlay = orig_img.copy()
            cv2.drawContours(overlay, [c], -1, tumor_color, -1)
            cv2.addWeighted(overlay, 0.4, orig_img, 0.6, 0, orig_img)
            cv2.drawContours(orig_img, [c], -1, tumor_color, 2)
        highlighted_filename = f"tumor_{filename}"
        highlighted_filepath = os.path.join(app.config['UPLOAD_FOLDER'], highlighted_filename)
        cv2.imwrite(highlighted_filepath, orig_img)
        return url_for('static', filename=f"uploads/{highlighted_filename}")
    except Exception as e:
        print(f"Error drawing highlight: {e}")
        return url_for('static', filename=f"uploads/{filename}")

# ---------------- ENTRY POINT ----------------
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

# ---------------- HOME / DASHBOARD ----------------
@app.route('/home')
def home():
    if 'user_id' not in session:
        flash("Please login to access the dashboard.")
        return redirect(url_for('login'))
    return render_template("index.html")

# ---------------- PREVIEW & PERFORMANCE ----------------
@app.route('/preview')
def preview():
    if 'user_id' not in session:
        flash("Login required.")
        return redirect(url_for('login'))
    return render_template("preview.html")

@app.route('/performance')
def performance():
    if 'user_id' not in session:
        flash("Login required.")
        return redirect(url_for('login'))
    return render_template("performance.html")

@app.route('/chart')
def chart():
    if 'user_id' not in session:
        flash("Login required.")
        return redirect(url_for('login'))
    
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    
    # Get total scans for this user
    cursor.execute("SELECT COUNT(*) as total FROM scans WHERE user_id = %s", (session['user_id'],))
    total = cursor.fetchone()['total']
    
    # Get tumor count
    cursor.execute("SELECT COUNT(*) as count FROM scans WHERE user_id = %s AND result LIKE '%Tumor Detected%'", (session['user_id'],))
    tumor_count = cursor.fetchone()['count']
    
    # Get no tumor count
    no_tumor_count = total - tumor_count
    
    # Get latest scan date
    cursor.execute("SELECT scan_date FROM scans WHERE user_id = %s ORDER BY scan_date DESC LIMIT 1", (session['user_id'],))
    latest = cursor.fetchone()
    latest_scan = latest['scan_date'] if latest else "No scans yet"
    
    cursor.close()
    conn.close()
    
    return render_template("chart.html", 
                           tumor_count=tumor_count,
                           no_tumor_count=no_tumor_count,
                           total=total,
                           latest_scan=latest_scan)

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("Login required.")
        return redirect(url_for('login'))
        
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM scans WHERE user_id = %s ORDER BY scan_date DESC', (session['user_id'],))
    scans = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template("history.html", scans=scans)

@app.route('/download_report/<int:scan_id>')
def download_report(scan_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM scans WHERE id = %s AND user_id = %s', (scan_id, session['user_id']))
    scan = cursor.fetchone()
    cursor.close()
    conn.close()

    if not scan:
        flash("Scan not found.")
        return redirect(url_for('history'))

    # Create PDF Report
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Brain Tumor Detection - Diagnostic Report", 0, 1, 'C')
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", '', 12)
    # Ensure patient name has no emojis just in case
    clean_name = scan['patient_name'].encode('latin-1', 'replace').decode('latin-1')
    pdf.cell(50, 10, f"Patient Name: {clean_name}", 0, 1)
    pdf.cell(50, 10, f"Scan Date: {scan['scan_date']}", 0, 1)
    pdf.ln(5)
    
    # Remove emojis from result string
    clean_result = scan['result'].replace('⚠️', '').replace('✅', '').replace('🚫', '').strip()
    
    # Diagnostics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(50, 10, f"Result: {clean_result}", 0, 1)
    pdf.cell(50, 10, f"Confidence: {scan['confidence']}%", 0, 1)
    pdf.ln(10)
    
    # Embed image if it exists
    if scan['image_url']:
        img_path = 'c:/Users/Deepti/Desktop/brain-tumor-project' + scan['image_url']
        try:
            pdf.image(img_path, w=100)
        except Exception as e:
            pdf.cell(50, 10, "(Image could not be loaded into PDF)", 0, 1)

    pdf.ln(20)

    # Output to Memory
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"Medical_Report_{scan['patient_name']}.pdf"
    )

@app.route('/delete_scan/<int:scan_id>', methods=['POST'])
def delete_scan(scan_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()
    # Only delete if the scan belongs to the logged-in user
    cursor.execute('DELETE FROM scans WHERE id = %s AND user_id = %s', (scan_id, session['user_id']))
    conn.commit()
    cursor.close()
    conn.close()

    flash("Scan deleted successfully.")
    return redirect(url_for('history'))


# ---------------- AUTH ROUTES ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, hashed_password))
            conn.commit()
            cursor.close()
            conn.close()
            flash("Registration successful! Please login.")
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            if err.errno == 1062: # Duplicate entry
                flash("Username already exists. Try another.")
            else:
                flash("Database error occurred.")

            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

            
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash("Welcome back!")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for('index'))

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash("Unauthorized. Please login again.")
        return redirect(url_for('login'))
    
    if 'image' not in request.files:
        flash("No image uploaded.")
        return redirect(url_for('preview'))
    
    file = request.files['image']
    if file.filename == '':
        flash("No selected file.")
        return redirect(url_for('preview'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess Image
        img = cv2.imread(filepath)
        if img is None:
            flash("Invalid image format.")
            return redirect(url_for('preview'))
            
        # Optional Image Validation
        patient_name = request.form.get('patient_name', 'Anonymous Patient').strip() or "Anonymous Patient"
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_url = url_for('static', filename=f"uploads/{filename}")

        if not is_brain_mri(img):
            return render_template("result.html", 
                               result="Invalid Scan Detected 🚫", 
                               confidence=0, 
                               location=None, 
                               desc="The uploaded image does not appear to be a standard Brain MRI. Please upload a valid Brain MRI scan.", 
                               image=img_url, 
                               patient_name=patient_name, 
                               date=date)
        
        # Get the highest accuracy model available
        if 'multiclass' in models:
            model = models.get('multiclass')
            model_name = "Multi-Class MobileNetV2"
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (128, 128))
            img_final = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_resized))
            img_final = np.expand_dims(img_final, axis=0)
            
            preds = model.predict(img_final)[0]
            class_idx = np.argmax(preds)
            confidence = round(float(preds[class_idx] * 100), 2)
            
            # Map index to class names (assuming standard Kaggle dataset order)
            # 0: Glioma, 1: Meningioma, 2: No Tumor, 3: Pituitary
            class_map = {
                0: "Glioma Tumor ⚠️",
                1: "Meningioma Tumor ⚠️",
                2: "No Tumor Detected ✅",
                3: "Pituitary Tumor ⚠️"
            }
            
            result = class_map.get(class_idx, "Unknown Type")
            
            if class_idx != 2: # If it is a tumor
                location = "Brain Section"
                if class_idx == 0:
                    desc = "Glioma detected. These tumors start in the glial cells that surround nerve cells."
                elif class_idx == 1:
                    desc = "Meningioma detected. This tumor arises from the meninges — the membranes that surround your brain."
                else:
                    desc = "Pituitary tumor detected. These are abnormal growths that develop in your pituitary gland."
                
                img_url = draw_highlight(filepath, filename)
            else:
                location = None
                desc = "Scan appears clear. No significant abnormalities."

        elif 'efficientnet' in models:
            model = models.get('efficientnet')
            model_name = "EfficientNetB0"
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_final = tf.keras.applications.efficientnet.preprocess_input(np.array(img_resized))
            img_final = np.expand_dims(img_final, axis=0)
            pred = model.predict(img_final)[0][0]
            if pred > 0.5:
                result = "Tumor Detected ⚠️"
                confidence = round(float(pred * 100), 2)
                location = "Brain Section"
                desc = "Abnormal tissue growth detected."
                img_url = draw_highlight(filepath, filename)
            else:
                result = "No Tumor Detected ✅"
                confidence = round(float((1 - pred) * 100), 2)
                location = None
                desc = "Scan clear."
        
        elif 'mobilenet' in models:
            model = models.get('mobilenet')
            model_name = "MobileNetV2"
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (128, 128))
            img_final = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_resized))
            img_final = np.expand_dims(img_final, axis=0)
            
            pred = model.predict(img_final)[0][0]
            if pred > 0.5:
                result = "Tumor Detected ⚠️"
                confidence = round(float(pred * 100), 2)
                location = "Brain Section" 
                desc = "Abnormal tissue growth detected in the MRI scan."
                # Draw Highlight
                img_url = draw_highlight(filepath, filename)
            else:
                result = "No Tumor Detected ✅"
                confidence = round(float((1 - pred) * 100), 2)
                location = None
                desc = "Scan appears clear."
        
        elif 'resnet50' in models:
            model = models.get('resnet50')
            model_name = "ResNet50"
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_final = tf.keras.applications.resnet50.preprocess_input(np.array(img_resized).astype('float32'))
            img_final = np.expand_dims(img_final, axis=0)
            pred = model.predict(img_final)[0][0]
            if pred > 0.5:
                result = "Tumor Detected ⚠️"
                confidence = round(float(pred * 100), 2)
                location = "Brain Section"
                desc = "Abnormal growth detected."
                img_url = draw_highlight(filepath, filename)
            else:
                result = "No Tumor Detected ✅"
                confidence = round(float((1 - pred) * 100), 2)
                location = None
                desc = "Scan clear."
        else:
            result = "Model Not Loaded"
            confidence = 0
            location = None
            desc = ""

        # Save to History
        try:
            conn = get_db()
            cursor = conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO scans (user_id, patient_name, scan_date, result, confidence, image_url)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (session['user_id'], patient_name, timestamp, result, confidence, img_url))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error saving to history: {e}")

        return render_template("result.html", 
                               result=result, 
                               confidence=confidence, 
                               location=location, 
                               desc=desc, 
                               image=img_url, 
                               patient_name=patient_name, 
                               date=date,
                               model_used=model_name)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)