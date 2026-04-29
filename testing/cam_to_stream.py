import cv2
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "0.0.0.0"
PORT = 8080
FRAMESIZE = (320,240)

webcam = cv2.VideoCapture(0)

def generate_frames():
    ret, frame = webcam.read()
    if not ret:
        return None
    
    frame = cv2.resize(frame, FRAMESIZE)
    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve a simple HTML page with the stream embedded
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body style="margin:0; background:#000;">
                    <img src="/stream" style="width:100%; height:100vh; object-fit:contain;">
                </body>
                </html>
            """)

        elif self.path == '/stream':
            # Serve the MJPEG stream
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    frame = generate_frames()
                    if frame is None:
                        break
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected

    def log_message(self, format, *args):
        pass  # Suppress default request logs

if __name__ == '__main__':
    server = HTTPServer((HOST, PORT), StreamHandler)
    print(f"Streaming at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        webcam.release()
        server.server_close()