import requests
import sys
import os

def run_tests():
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    # 1. Register a new user
    print("Testing Registration...")
    reg_data = {
        "username": "pres_user_test",
        "password": "password123"
    }
    r = session.post(f"{base_url}/register", data=reg_data, allow_redirects=True)
    if "Registration successful" in r.text or "Username already exists" in r.text:
        print("Registration flow: PASS")
    else:
        print("Registration flow: FAIL (Check logs)")
        print(r.text[:500])
        sys.exit(1)
        
    # 2. Login
    print("\nTesting Login...")
    login_data = {
        "username": "pres_user_test",
        "password": "password123"
    }
    r = session.post(f"{base_url}/login", data=login_data, allow_redirects=True)
    if "Welcome to Brain Tumor Detection" in r.text or "Welcome back!" in r.text:
        print("Login flow: PASS")
    else:
        print("Login flow: FAIL")
        print(r.text[:500])
        sys.exit(1)

    # 3. Test predicting a valid tumor MRI
    print("\nTesting Tumor MRI Prediction...")
    image_path = "c:/Users/Deepti/Desktop/brain-tumor-project/data_split/test/tumor/Tr-gl_1005.jpg"
    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}")
        sys.exit(1)
        
    with open(image_path, "rb") as img_file:
        files = {
            "image": ("Tr-gl_1005.jpg", img_file, "image/jpeg")
        }
        data = {
            "patient_name": "E2E Test Patient"
        }
        r = session.post(f"{base_url}/predict", data=data, files=files, allow_redirects=True)
        
    if "Result" in r.text or "Tumor" in r.text:
        print("Tumor Prediction flow: PASS")
        if "Tumor Detected" in r.text or "Glioma" in r.text or "Meningioma" in r.text or "Pituitary" in r.text:
            print("  Result correctly identified tumor presence.")
        else:
            print("  Warning: Result text did not contain expected tumor message.")
    else:
        print("Predict flow: FAIL")
        print(r.text[:500])
        sys.exit(1)

    # 4. Test predicting an invalid image (color doctor background)
    print("\nTesting Invalid Image Detection...")
    invalid_image_path = "c:/Users/Deepti/Desktop/brain-tumor-project/static/doctor_bg.png"
    if os.path.exists(invalid_image_path):
        with open(invalid_image_path, "rb") as img_file:
            files = {
                "image": ("doctor_bg.png", img_file, "image/png")
            }
            data = {
                "patient_name": "E2E Invalid Patient"
            }
            r = session.post(f"{base_url}/predict", data=data, files=files, allow_redirects=True)
            
        if "Invalid Scan Detected" in r.text or "does not appear to be a standard Brain MRI" in r.text:
            print("Invalid Image Detection flow: PASS")
        else:
            print("Invalid Image Detection flow: FAIL")
            print(r.text[:500])
            sys.exit(1)
    else:
        print("doctor_bg.png not found, skipping invalid image check.")

    # 5. Check History page
    print("\nTesting History Page...")
    r = session.get(f"{base_url}/history")
    if "E2E Test Patient" in r.text:
        print("History page shows test record: PASS")
    else:
        print("History page check: FAIL")
        sys.exit(1)

    # 6. Check Chart page
    print("\nTesting Chart Page...")
    r = session.get(f"{base_url}/chart")
    if "total" in r.text or "Scans Analyzed" in r.text:
        print("Chart page renders: PASS")
    else:
        print("Chart page check: FAIL")
        sys.exit(1)

    # 7. Check Performance page
    print("\nTesting Performance Page...")
    r = session.get(f"{base_url}/performance")
    if "Performance Report" in r.text:
        print("Performance page renders: PASS")
    else:
        print("Performance page check: FAIL")
        sys.exit(1)

    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
