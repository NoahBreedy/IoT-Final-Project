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


# Manages data from a single result from a stream
class StrmResult:
    def __init__(self, yolo:Results, aruco_corners=None, aruco_ids=None):
        # YOLO Results
        self.yolo:Results = yolo
        # ARUCO Results
        self.arcuo_corners = aruco_corners
        self.arcuo_corners = aruco_ids
        
        self.output_frame = None
        self.output_jpeg  = None
        self.input_frame  = self.yolo.orig_img
        self.input_jpeg   = None
        self.lock = threading.Lock()
    
    def _makeOutFrame(self):
        self.output_frame = self.yolo.plot(show=False)
    
    def _makeInFrame(self):
        self.input_frame = self.yolo.orig_img

    def _makeOutJPEG(self):
        if self.output_frame is None: self._makeOutFrame()
        _, self.output_jpeg = cv2.imencode('.jpg', self.output_frame)

    def _makeInJPEG(self):
        if self.input_frame is None: self._makeOutFrame()
        _, self.input_jpeg = cv2.imencode('.jpg', self.input_frame)

    def getOutFrame(self):
        with self.lock:
            if self.output_frame is None: self._makeOutFrame()
            return self.output_frame

    def getInFrame(self):
        with self.lock:
            if self.input_frame is None: self._makeInFrame()
            return self.input_frame

    def getOutJPEG(self):
        with self.lock:
            if self.output_jpeg is None: self._makeOutJPEG()
            return self.output_jpeg

    def getInJPEG(self):
        with self.lock:
            if self.input_jpeg is None: self._makeInJPEG()
            return self.input_jpeg



class StreamData:
    def __init__(self, src_url:str):
        self.src_url = src_url
        self.in_stream = cv2.VideoCapture(self.src_url)
        self.in_stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.latest_frame = None # Never gets reset to None, always the most recent frame
        self.latest_jpeg = None
        self.in_frame = None
        self.in_lock = threading.Lock()
        self.in_thread = threading.Thread(target=StreamData.readLoop, daemon=True, args=[self])
        self.in_thread.start()

        self.result:StrmResult = None
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
                self.in_frame     = new_frame
                self.latest_frame = new_frame
                self.latest_jpeg  = None
    
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
                self.result = StrmResult(
                    tmp_results,
                    None, # Placeholders for aruco stuff
                    None,
                )
            
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
    def getResult(self, new:bool=True):
        if not self.wait():
            return None
        with self.out_lock:
            return self.result
    
    # Gets the most recent frame from the stream
    def getFrame(self):
        with self.in_lock:
            return self.latest_frame
    
    # Gets the most recent JPEG from the stream
    def getJPEG(self):
        with self.in_lock:
            if self.latest_jpeg is None:
                _, self.latest_jpeg = cv2.imencode('.jpg', self.latest_frame)
            return self.latest_jpeg
