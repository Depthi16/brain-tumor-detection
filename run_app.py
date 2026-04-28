import webview
import threading
import sys
import time

# Import the Flask app instance from your app.py
from app import app

def start_server():
    # Run the Flask app on localhost; disable reloader as we are in a wrapper
    app.run(host='127.0.0.1', port=5000, threaded=True, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start Flask API in a background thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Sleep slightly to let the server boot up
    time.sleep(1)

    # Create the Desktop GUI Window pointing to the local Flask application
    window = webview.create_window(
        'Brain Tumor Analytics & Detection App', 
        'http://127.0.0.1:5000/',
        width=1280,
        height=850,
        min_size=(900, 650)
    )
    
    # Start the application loop
    webview.start()
    
    # Once the window is closed, shut down the script entirely
    sys.exit()
