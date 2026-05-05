import threading
from time import sleep

from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import cv2


# =========================
# YOLO MODEL
# =========================
model = YOLO("yolo11n.pt")
model_lock = threading.Lock()
class_names = model.names

# Thread-safe prediction function
def predict(frame):
    with model_lock:
        results = model(frame, verbose=False)
        result:Results = results[0]
        return result

# =========================
# ARUCO SETUP
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
MARKER_SIZE = 0.03  # Size in meters // 3 cm
DIST_THRESHOLD = .1


# =========================
# CAMERA CALIBRATION
# =========================
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=float)


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
            tmp_results = predict(tmp_frame)
            
            # Set output vars
            with self.out_lock:
                self.results = tmp_results
                self.out_frame = None
                self.out_jpeg = None
            
            # Wake up wating threads
            self.event.set()
            self.event.clear()

    # Wait for a new frame
    def wait(self, timeout:float=5.0): 
        ret = self.event.wait(timeout=timeout)
        if not ret:
            print(f"WARNING: Timed out while waiting for prediction {self.src_url}")
        return ret
        
    # Get the entire results object from a new prediction
    def getResults(self):
        if not self.wait():
            return None
        with self.out_lock:
            return self.results
    
    # Get the numpy image from a new prediction
    def getOutFrame(self):
        if not self.wait():
            return None
        with self.out_lock:
            if self.out_frame is None:
                self.out_frame = self.results.plot(show=False)
            return self.out_frame
    
    # Get the JPEG image from a new prediction
    def getOutJPEG(self):
        if not self.wait():
            return None
        with self.out_lock:
            if self.out_frame is None:
                self.out_frame = self.results.plot(show=False)
            if self.out_jpeg is None:
                _, self.out_jpeg = cv2.imencode('.jpg', self.out_frame)
            return self.out_jpeg
    
    # Get the numpy image used as input:
        # latest=True  --> Get the most recent capture
        # latest=False --> Get the input for a new prediciton (waits for new prediction)
    def getInFrame(self, latest:bool=True):
        if latest:
            with self.in_lock:
                return self.in_frame
        else:
            if not self.wait():
                return None
            with self.out_lock:
                return self.results.orig_img
    
    
    # Get the JPEG image used as input:
        # latest=True  --> Get the most recent capture
        # latest=False --> Get the input for a new prediciton (waits for new prediction)
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