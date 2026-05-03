from ultralytics import YOLO
import cv2
import cv2.aruco as aruco

from time import sleep
import threading

from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "0.0.0.0"
PORT = 9080

model = YOLO("yolo11n.pt")
model_lock = threading.Lock()

def predict(frame):
    with model_lock:
        results = model(frame, verbose=False)
        return results[0].plot(show=False)


class StreamData:
    def __init__(self, src_url:str):
        self.src_url = src_url
        self.in_stream = cv2.VideoCapture(self.src_url)
        self.in_stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.in_frame = None
        self.in_lock = threading.Lock()
        self.in_thread = threading.Thread(target=StreamData.readLoop, daemon=True, args=[self])
        self.in_thread.start()

        self.jpeg = None
        self.out_frame = None
        self.out_lock  = threading.Lock()
        self.event = threading.Event()
        self.predict_thread = threading.Thread(target=StreamData.predictLoop, daemon=True, args=[self])
        self.predict_thread.start()

    # Reads new images from the stream
    def readLoop(self):
        while True:
            ret, new_frame = self.in_stream.read()
            if not ret: 
                continue
            
            # Update input frame
            with self.in_lock:
                self.in_frame = new_frame
    
    # Makes predictions on the newest images
    def predictLoop(self):
        while True:
            sleep(1/120) # Give some guarenteed sleep time for readLoop to get a new frame
            if self.in_frame is None:
                sleep(1/30)
                continue

            # Grab the input frame (only block while retrieving)
            tmp_frame = None
            with self.in_lock:
                tmp_frame = self.in_frame
                self.in_frame = None

            # Make prediction
            prediction = predict(tmp_frame)            
            if prediction is None:
                continue
            
            # Encode and set output
            with self.out_lock:
                self.out_frame = prediction
                _, self.jpeg = cv2.imencode('.jpg', self.out_frame)
            
            # Wake up wating threads
            self.event.set()
            self.event.clear()

    def wait(self): return self.event.wait(timeout=1.0)

    def get(self):
        if not self.wait():
            print("WARNING: Timed out while waiting for updated image")
            return None

        with self.out_lock:
            return self.jpeg


stream_data:dict[StreamData] = {
    "0": StreamData("http://localhost:8080/stream/0"),
    "1": StreamData("http://localhost:8080/stream/1"),
    "2": StreamData("http://localhost:8080/stream/2"),
    "3": StreamData("http://localhost:8080/stream/3"),
    "4": StreamData("http://localhost:8080/stream/4"),
}

# Main http server's stream handler
class StreamHandler(BaseHTTPRequestHandler):
    STREAM_DIR = "/prediction/"

    def invalidAddress(self):
        self.wfile.write(b"Yeah you typed in the wrong address, sorry.\r\n")
        self.wfile.write(f"Try something like: \"http://IP{StreamHandler.STREAM_DIR}X\", where X is the stream's ID\r\n".encode())
    
    def do_GET(self):
        if self.path[0:len(self.STREAM_DIR)] == self.STREAM_DIR:
            # Validation of address
            index_str = self.path[len(self.STREAM_DIR):]
            stream = None
            try:
                stream = stream_data[index_str]
            except:
                self.invalidAddress()
                return

            # Address is valid after this point

            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            try:
                while True:
                    # Get webcam image
                    jpeg = stream.get()
                    if jpeg is None: continue
                    # Write image data
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                print("Client disconnected")
        else:
            self.invalidAddress()
        
    
    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    # Create server
    server = ThreadedHTTPServer((HOST,PORT), StreamHandler)

    # Each client gets a thread
    server.socket.setsockopt(__import__('socket').SOL_SOCKET, __import__('socket').SO_REUSEADDR, 1)
    print(f"Streaming at \"http://IP:{PORT}\"")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close()



if __name__ == "__main__":
    main()