import mysql.connector
from werkzeug.security import generate_password_hash

def reset_password(username, new_password):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="brain_tumor_db"
        )
        cursor = conn.cursor()
        hashed_password = generate_password_hash(new_password)
        cursor.execute("UPDATE users SET password = %s WHERE username = %s", (hashed_password, username))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Password for user '{username}' has been reset to '{new_password}'.")
        else:
            print(f"User '{username}' not found.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    reset_password("depthi", "admin123")
