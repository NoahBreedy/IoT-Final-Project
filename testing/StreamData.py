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
def predictYOLO(frame):
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
aruco_lock = threading.Lock()

dist_coeffs = np.zeros((5, 1))
MARKER_SIZE = 0.03  # 2 cm

# Thread-safe aruco prediction function
def predictARUCO(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    with aruco_lock:
        ar_corners, ar_ids, _ = aruco_detector.detectMarkers(gray)

    if ar_ids is None: return None, None, None

    ar_rvecs, ar_tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        ar_corners, MARKER_SIZE, camera_matrix, dist_coeffs
    )

    ar_detected_markers = []

    for i, corner in enumerate(ar_corners):
        pts = corner[0].astype(int)

        marker_id = int(ar_ids[i][0])

        # Center of marker
        cX = int(np.mean(pts[:, 0]))
        cY = int(np.mean(pts[:, 1]))

        # Distance (forward distance is best)
        distance = ar_tvecs[i][0][2]

        # Save for YOLO matching
        ar_detected_markers.append((marker_id, cX, cY, distance))

    return ar_rvecs, ar_tvecs, ar_detected_markers


# =========================
# CAMERA CALIBRATION
# =========================
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=float)


# Manages data from a single result from a stream
class StrmResult:
    def __init__(self, yolo:Results, aruco_detected_markers=None, aruco_rvecs=None, aruco_tvecs=None):
        # YOLO Results
        self.yolo:Results = yolo
        # ARUCO Results
        self.aruco_detected_markers = aruco_detected_markers
        self.aruco_rvecs = aruco_rvecs
        self.aruco_tvecs = aruco_tvecs
        
        self.output_frame = None
        self.output_jpeg  = None
        self.input_frame  = self.yolo.orig_img
        self.input_jpeg   = None
        self.lock = threading.Lock()
    
    def _makeOutFrame(self, yolo:bool=True, aruco:bool=True):
        if yolo:
            self.output_frame = self.yolo.plot(show=False)
        else:
            self.output_frame = self.input_frame.copy()
        
        if aruco and self.aruco_detected_markers is not None:
            for i in range(len(self.aruco_detected_markers)):
                marker_id, cX, cY, distance = self.aruco_detected_markers[i]

                # Draw axis (helps visualize pose)
                cv2.drawFrameAxes(self.output_frame, camera_matrix, dist_coeffs,
                                self.aruco_rvecs[i], self.aruco_tvecs[i], 0.03)

                # Show ID + distance
                cv2.putText(self.output_frame,
                            f"ID:{marker_id} {distance:.2f}m",
                            (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
        else:
            pass
    
    # Makes the input numpy frame (NOT THREAD SAFE)
    def _makeInFrame(self):
        self.input_frame = self.yolo.orig_img

    # Makes the output numpy frame (NOT THREAD SAFE)
    def _makeOutJPEG(self):
        if self.output_frame is None: self._makeOutFrame()
        _, self.output_jpeg = cv2.imencode('.jpg', self.output_frame)

    # Makes the input JPEG image
    def _makeInJPEG(self):
        if self.input_frame is None: self._makeOutFrame()
        _, self.input_jpeg = cv2.imencode('.jpg', self.input_frame)

    # Returns the output numpy frame (with predictions plotted)
    def getOutFrame(self):
        with self.lock:
            if self.output_frame is None: self._makeOutFrame()
            return self.output_frame

    # Returns the input numpy frame
    def getInFrame(self):
        with self.lock:
            if self.input_frame is None: self._makeInFrame()
            return self.input_frame

    # Returns the output jpeg (with predictions plotted)
    def getOutJPEG(self):
        with self.lock:
            if self.output_jpeg is None: self._makeOutJPEG()
            return self.output_jpeg

    # Returns the input jpeg
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

            # Make YOLO prediction
            yolo_results = predictYOLO(tmp_frame)
            # Make ARUCO prediction
            ar_rvecs, ar_tvecs, ar_detected_markers = predictARUCO(tmp_frame)
            
            # Set output vars
            with self.out_lock:
                self.result = StrmResult(
                    yolo_results,
                    ar_detected_markers,
                    ar_rvecs,
                    ar_tvecs
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
        if new and not self.wait():
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
