import mysql.connector
import io
from fpdf import FPDF
import sys

def test_pdf(scan_id):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="brain_tumor_db"
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM scans WHERE id = %s', (scan_id,))
        scan = cursor.fetchone()
        cursor.close()
        conn.close()

        if not scan:
            print(f"Scan {scan_id} not found.")
            return

        print("Scan data:", scan)

        # Create PDF Report
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Brain Tumor Detection - Diagnostic Report", 0, 1, 'C')
        pdf.ln(10)
        
        # Patient Info
        pdf.set_font("Arial", '', 12)
        clean_name = scan['patient_name'].encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(50, 10, f"Patient Name: {clean_name}", 0, 1)
        pdf.cell(50, 10, f"Scan Date: {scan['scan_date']}", 0, 1)
        pdf.ln(5)
        
        # Remove emojis from result string
        clean_result = scan['result'].replace('⚠️', '').replace('✅', '').replace('🚫', '').strip()
        print(f"Original result: {repr(scan['result'])}")
        print(f"Cleaned result: {repr(clean_result)}")
        
        # Diagnostics
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, f"Result: {clean_result}", 0, 1)
        pdf.cell(50, 10, f"Confidence: {scan['confidence']}%", 0, 1)
        pdf.ln(10)
        
        # Embed image if it exists
        if scan['image_url']:
            img_path = 'c:/Users/Deepti/Desktop/brain-tumor-project' + scan['image_url']
            print("Image path:", img_path)
            try:
                pdf.image(img_path, w=100)
                print("Image loaded successfully.")
            except Exception as e:
                print(f"Error loading image: {e}")
                pdf.cell(50, 10, "(Image could not be loaded into PDF)", 0, 1)

        pdf.ln(20)

        # Output to Memory
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        print("Success! PDF bytes length:", len(pdf_bytes))
    except Exception as e:
        print("Error generated during test:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    scan_id = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    test_pdf(scan_id)
