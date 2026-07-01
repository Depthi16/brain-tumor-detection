import mysql.connector

def check_schema():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="brain_tumor_db"
        )
        cursor = conn.cursor()
        cursor.execute("DESCRIBE users")
        schema = cursor.fetchall()
        print("Users Table Schema:")
        for field in schema:
            print(field)
        
        cursor.execute("DESCRIBE scans")
        schema = cursor.fetchall()
        print("\nScans Table Schema:")
        for field in schema:
            print(field)
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schema()
