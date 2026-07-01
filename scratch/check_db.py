import mysql.connector

def check_users():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="brain_tumor_db"
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, username, password FROM users")
        users = cursor.fetchall()
        print("Users in database:")
        for user in users:
            print(f"ID: {user['id']}, Username: {user['username']}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_users()
