from ultralytics import YOLO
from ultralytics.engine.results import Results
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
        result:Results = results[0]
        return result.plot(show=False)


class StreamData:
    def __init__(self, src_url:str):
        self.src_url = src_url
        self.in_stream = cv2.VideoCapture(self.src_url)
        self.in_stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.in_frame = None
        self.in_jpeg = None
        self.in_lock = threading.Lock()
        self.in_thread = threading.Thread(target=StreamData.readLoop, daemon=True, args=[self])
        self.in_thread.start()

        self.out_jpeg = None
        self.out_frame = None
        self.results:Results = None
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
                self.in_jpeg = None
    
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
            with model_lock:
                results = model(tmp_frame, verbose=False)
                self.results = results[0]
            
            # Set output vars
            with self.out_lock:
                self.out_frame = None
                self.out_jpeg = None
            
            # Wake up wating threads
            self.event.set()
            self.event.clear()

    def wait(self): 
        ret = self.event.wait(timeout=5.0)
        if not ret:
            print(f"WARNING: Timed out while waiting for prediction {self.src_url}")
        return ret

    def getAll(self):
        if not self.wait(): return None
        with self.out_lock:
            return (self.out_jpeg, self.out_frame, self.results)
        
    def getResults(self):
        if not self.wait():
            return None
        with self.out_lock:
            return self.results
    
    def getOutJPEG(self):
        if not self.wait():
            return None
        with self.out_lock:
            if self.out_frame is None:
                self.out_frame = self.results.plot(show=False)
            if self.out_jpeg is None:
                _, self.out_jpeg = cv2.imencode('.jpg', self.out_frame)
            return self.out_jpeg
    
    def getOutFrame(self):
        if not self.wait():
            return None
        with self.out_lock:
            if self.out_frame is None:
                self.out_frame = self.results.plot(show=False)
            return self.out_frame
    
    def getInFrame(self, latest:bool=True):
        if latest:
            with self.in_lock:
                return self.in_frame
        else:
            if not self.wait():
                return None
            with self.out_lock:
                return self.results.orig_img
    
    def getInJPEG(self, latest:bool=True):
        if latest:
            with self.in_lock:
                if self.in_frame is None: return None
                if self.in_jpeg is None:
                    _, self.in_jpeg =  cv2.imencode('.jpg', self.in_frame)
                return self.in_jpeg
        else:
            if not self.wait():
                return None
            with self.out_lock:
                if self.out_jpeg is None:
                    _, self.out_jpeg =  cv2.imencode('.jpg', self.results.orig_img)
                return self.out_jpeg


stream_data:dict[StreamData] = {
    "0": StreamData("http://172.20.10.9:81/stream"),
    "1": StreamData("http://172.20.10.8:81/stream"),
    "2": StreamData("http://172.20.10.7:81/stream"),
}

# Main http server's stream handler
class StreamHandler(BaseHTTPRequestHandler):
    PRED_DIR = "/prediction/"
    RESTREAM_DIR = "/stream/"
    DIRECT_STREAM_DIR = "/direct/"

    def invalidAddress(self):
        self.wfile.write(b"Yeah you typed in the wrong address, sorry.\r\n")
        self.wfile.write(f"Try something like: \"http://IP{StreamHandler.PRED_DIR}X\", where X is the stream's ID\r\n".encode())
    
    def startStream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

    def sendJPEG(self, jpeg):
        # Write image data
        self.wfile.write(b"--frame\r\n")
        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
        self.wfile.write(jpeg)
        self.wfile.write(b"\r\n")

    def stream(self, stream_type:int, stream_id:str):
        # Validation of address
        stream:StreamData = None
        try:
            stream = stream_data[stream_id]
        except:
            self.invalidAddress()
            return
        # Address is valid after this point

        self.startStream()

        while True:
            # Get data based on stream type
            if stream_type == 1:
                jpeg = stream.getOutJPEG()
            elif stream_type == 2:
                jpeg = stream.getInJPEG(latest=False)
            elif stream_type == 3:
                jpeg = stream.getInJPEG(latest=True)
            else:
                print(f"ERROR: Invalid stream_type: {stream_type}")
                return

            # Send stream data
            if jpeg is None:
                continue
            self.sendJPEG(jpeg)

    def do_GET(self):
        print(f"Client connecting at path {self.path}")
        stream_type = None
        stream_id = None

        if self.path[0:len(self.PRED_DIR)] == self.PRED_DIR: 
            stream_id = self.path[len(self.PRED_DIR):]
            stream_type = 1
        if self.path[0:len(self.RESTREAM_DIR)] == self.RESTREAM_DIR: 
            stream_id = self.path[len(self.RESTREAM_DIR):]
            stream_type = 2
        if self.path[0:len(self.DIRECT_STREAM_DIR)] == self.DIRECT_STREAM_DIR: 
            stream_id = self.path[len(self.DIRECT_STREAM_DIR):]
            stream_type = 3

        try:
            if stream_type is not None:
                self.stream(stream_type, stream_id)
            else:
                self.invalidAddress()
        except (BrokenPipeError, ConnectionResetError):
            print(f"Client disconnected from path {self.path}")
            pass

        
    
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