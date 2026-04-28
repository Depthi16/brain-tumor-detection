import os
import cv2
import numpy as np
from flask import Flask, render_template, request, url_for, session, redirect, flash, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
import io
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
if os.path.exists("mobilenet_model.h5"):
    models['mobilenet'] = load_model("mobilenet_model.h5")
    print("Model mobilenet loaded successfully from mobilenet_model.h5.")
else:
    print("Warning: Model not found at mobilenet_model.h5")

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
        
        # Get the MobileNet model
        model = models.get('mobilenet')

        # Preprocess Image (128x128 for MobileNet)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        img_final = np.array(img_resized) / 255.0
        img_final = np.expand_dims(img_final, axis=0)

        # Predict
        if model:
            pred = model.predict(img_final)[0][0]
            # categories: 0: tumor, 1: no_tumor
            if pred < 0.5:
                result = "Tumor Detected ⚠️"
                confidence = round(float((1 - pred) * 100), 2)
                location = "Brain Section" 
                desc = "Abnormal tissue growth detected in the MRI scan. See the highlighted region."
                
                # Highlight the tumor in the original image using OpenCV
                try:
                    orig_img = cv2.imread(filepath)
                    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    max_val = np.max(blur)
                    # Adaptive thresholding: try to catch the brightest region which is often the tumor
                    threshold_value = max(100, max_val * 0.7) 
                    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
                    
                    kernel = np.ones((5,5), np.uint8)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) > 30: # Slightly lower threshold for area
                            # Create an orange color for the tumor (BGR format in OpenCV)
                            tumor_color = (100, 150, 255) # Light orange/peach
                            
                            # Draw an orange filled translucent overlay over the tumor area
                            overlay = orig_img.copy()
                            cv2.drawContours(overlay, [c], -1, tumor_color, -1)
                            cv2.addWeighted(overlay, 0.4, orig_img, 0.6, 0, orig_img)
                            
                            # Draw a solid orange boundary line
                            cv2.drawContours(orig_img, [c], -1, tumor_color, 2)
                            
                            # Find the center of the tumor to point the line
                            M = cv2.moments(c)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                            else:
                                x, y, w, h = cv2.boundingRect(c)
                                cX, cY = x + w // 2, y + h // 2
                                
                            # Calculate line start point (upper left from tumor)
                            start_x = max(10, cX - 60)
                            start_y = max(20, cY - 80)
                            
                            # Draw the pointing line
                            cv2.line(orig_img, (start_x, start_y), (cX, cY), (255, 255, 255), 1)
                            
                            # Add the "BRAIN TUMOR" label text
                            text_x = max(5, start_x - 40)
                            text_y = max(15, start_y - 5)
                            cv2.putText(orig_img, "BRAIN TUMOR", (text_x, text_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    highlighted_filename = f"tumor_{filename}"
                    highlighted_filepath = os.path.join(app.config['UPLOAD_FOLDER'], highlighted_filename)
                    cv2.imwrite(highlighted_filepath, orig_img)
                    img_url = url_for('static', filename=f"uploads/{highlighted_filename}")
                except Exception as e:
                    print(f"Error drawing bounding box: {e}")
            else:
                result = "No Tumor Detected ✅"
                confidence = round(float(pred * 100), 2)
                location = None
                desc = "Scan appears clear. No significant abnormalities."
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
                               model_used=model_type.capitalize())

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)