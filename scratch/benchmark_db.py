import mysql.connector
import time

def test_conn(host):
    try:
        start = time.time()
        conn = mysql.connector.connect(
            host=host,
            user="root",
            password="root",
            database="brain_tumor_db"
        )
        conn.close()
        print(f"Connection to {host} succeeded in {time.time() - start:.4f} seconds")
    except Exception as e:
        print(f"Connection to {host} failed in {time.time() - start:.4f} seconds: {e}")

if __name__ == "__main__":
    test_conn("localhost")
    test_conn("127.0.0.1")
