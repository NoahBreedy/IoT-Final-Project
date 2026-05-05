import cv2
import numpy as np
from ultralytics import YOLO
import requests

# =========================
# ESP32 STREAM
# =========================


ARDUINO_URL_RIGHT = "http://172.20.10.7:81/move?dir=right"
ESP32_URL = "http://172.20.10.7:81/stream"
cap = cv2.VideoCapture(ESP32_URL)

# =========================
# YOLO MODEL
# =========================
model = YOLO("yolo11n.pt")
class_names = model.names

# =========================
# ARUCO SETUP
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# =========================
# CAMERA CALIBRATION (REPLACE WITH YOUR REAL VALUES)
# =========================
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=float)

dist_coeffs = np.zeros((5, 1))

# REAL marker size in meters (VERY IMPORTANT)
MARKER_SIZE = 0.03  # 2 cm

DIST_THRESHOLD = .1

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # ARUCO DETECTION
    # =========================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    detected_markers = []

    if ids is not None:
        # 🔥 POSE ESTIMATION (distance comes from here)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs
        )

        for i, corner in enumerate(corners):
            pts = corner[0].astype(int)

            # Draw marker outline
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

            marker_id = int(ids[i][0])

            # Center of marker
            cX = int(np.mean(pts[:, 0]))
            cY = int(np.mean(pts[:, 1]))

            # 🔥 DISTANCE (forward distance is best)
            distance = tvecs[i][0][2]

	    # =========================
	    # SEND COMMAND IF CLOSE
  	    # =========================
	    
            if distance < DIST_THRESHOLD:
                try:
                    requests.get(ARDUINO_URL_RIGHT)
                    print("Sent STOP command")
                except:
                    print("Failed to send command")

            # Save for YOLO matching
            detected_markers.append((marker_id, cX, cY, distance))

            # Draw axis (helps visualize pose)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvecs[i], tvecs[i], 0.03)

            # Show ID + distance
            cv2.putText(frame,
                        f"ID:{marker_id} {distance:.2f}m",
                        (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    # =========================
    # YOLO DETECTION
    # =========================
    results = model(frame, verbose=False)

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{class_names[cls_id]} {conf:.2f}"

            # =========================
            # MATCH ARUCO TO YOLO BOX
            # =========================
            aruco_text = "No ArUco"

            for marker_id, mx, my, dist in detected_markers:
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    aruco_text = f"Aruco:{marker_id} {dist:.2f}m"
                    break

            # Draw YOLO box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            full_label = f"{label} | {aruco_text}"

            (w, h), _ = cv2.getTextSize(full_label,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, 1)

            cv2.rectangle(frame, (x1, y1 - h - 5),
                          (x1 + w, y1), (0, 255, 0), -1)

            cv2.putText(frame, full_label,
                        (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow("YOLO + ArUco Distance (ESP32)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()