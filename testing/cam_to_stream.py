import cv2
import time
import threading
from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer


HOST = "0.0.0.0"
PORT = 8080
FRAMESIZE = (320,240)


# Shared data
class StreamData:
    def __init__(self, frame:bytes, update):
        self.frame    = frame
        self.__update = update
        self.lock  = threading.Lock()
        self.event = threading.Event()
    
    def update(self): 
        with self.lock:
            self.__update(self) # Update image data
        self.event.set()
        self.event.clear()

    def wait(self): return self.event.wait(timeout=1.0)

    def get(self):
        if not self.wait():
            print("ERROR: Timed out while waiting for webcam image")
            return None

        with self.lock:
            return self.frame

frame = None

def update_0(self:StreamData):
    subframe = cv2.resize(frame, FRAMESIZE)
    _, self.frame = cv2.imencode('.jpg', subframe)
    return subframe

def update_1(self:StreamData):
    subframe = frame[:frame.shape[0]//2, :frame.shape[1]//2]
    subframe = cv2.resize(subframe, FRAMESIZE)
    _, self.frame = cv2.imencode('.jpg', subframe)
    return subframe # Use if you want

def update_2(self:StreamData):
    subframe = frame[:frame.shape[0]//2, frame.shape[1]//2:]
    subframe = cv2.resize(subframe, FRAMESIZE)
    _, self.frame = cv2.imencode('.jpg', subframe)
    return subframe # Use if you want

def update_3(self:StreamData):
    subframe = frame[frame.shape[0]//2:, :frame.shape[1]//2]
    subframe = cv2.resize(subframe, FRAMESIZE)
    _, self.frame = cv2.imencode('.jpg', subframe)
    return subframe # Use if you want

def update_4(self:StreamData):
    subframe = frame[frame.shape[0]//2:, frame.shape[1]//2:]
    subframe = cv2.resize(subframe, FRAMESIZE)
    _, self.frame = cv2.imencode('.jpg', subframe)
    return subframe # Use if you want


stream_data:dict[StreamData] = {
    "0": StreamData(None, update_0),
    "1": StreamData(None, update_1),
    "2": StreamData(None, update_2),
    "3": StreamData(None, update_3),
    "4": StreamData(None, update_4),
}


# Capture frames endlessly; meant to run in another thread
def capture_frames():
    global frame
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to read next image from webcam")
            time.sleep(1)
            continue

        for path, stream in stream_data.items():
            stream.update()
        
        continue
    pass


def get_stream_content(id:str):
    if not id in stream_data.keys():
        print(f"ERROR: Stream id \"{id}\" not found")
        return
    
    stream:StreamData = stream_data[id]

    if not stream.wait():
        print("ERROR: Timed out while waiting for webcam image")
        return None

    with stream.lock:
        return stream.frame


# Main http server's stream handler
class StreamHandler(BaseHTTPRequestHandler):
    STREAM_DIR = "/stream/"
    STREAM_INDEX_MAX = 5

    def invalidAddress(self):
        self.wfile.write(b"Yeah you typed in the wrong address, sorry.\r\n")
        self.wfile.write(b"Try something like: \"http://IP/stream/X\", where X is the stream's ID\r\n")
    
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
    t = threading.Thread(target=capture_frames, daemon=True)
    t.start()

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


if __name__ == "__main__": main()