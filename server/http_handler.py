import requests
from time import sleep
from secrets import IPs
from enum import Enum

import numpy as np
import cv2

"""
Capture: Get jpeg from:   http://IP/capture
Stream:  Get stream from: http://IP:81/stream
Move:    Go to:           http://IP/move?dir=[forward/backwards/left/right/stop]
"""

class Dir(Enum):
    Stop=0,
    Forward=1,
    Backward=2,
    Left=3,
    Right=4


def dir_to_string(dir:Dir):
    if dir == Dir.Forward:
        return "forward"
    elif dir == Dir.Backward:
        return "backward"
    elif dir == Dir.Left:
        return "left"
    elif dir == Dir.Right:
        return "right"
    return "stop"

def get_capture_url(robot_id:int):
    if (0 <= robot_id < len(IPs)):
        return f"http://{IPs[robot_id]}/capture"
    return None

def get_capture_url(robot_id:int):
    if (0 <= robot_id < len(IPs)):
        return f"http://{IPs[robot_id]}:81/stream"
    return None

def get_move_url(robot_id:int, dir:Dir):
    if (0 <= robot_id < len(IPs)):
        dir = dir_to_string(dir)
        return f"http://{IPs[robot_id]}/move?dir={dir}"
    return None

def get_capture(robot_id):
    url = get_capture_url(robot_id)
    if url is None: return

    try:
        response = requests.get(url, timeout=3)
    except requests.exceptions.Timeout:
        print(f"WARINING: Connection to robot {robot_id} timed out: url=\"{url}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    return np.frombuffer(response.content, dtype=np.uint8)

def get_stream(robot_id:int):
    url = get_capture_url(robot_id)
    if url is None: return

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"WARNING: Failed to open cv2.VideoCapture from url \"{url}\"")
        return None

    return cap


def move(robot_id:int, dir:Dir):
    url = get_move_url(robot_id, dir)
    if url is None: return

    try:
        response = requests.get(url, timeout=3)
    except requests.exceptions.Timeout:
        print(f"WARINING: Connection to robot {robot_id} timed out: url=\"{url}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"MOVE! \"{url}\"")
    
    return 0 # Success

def main():
    ESC_KEY = 27
    video_stream = get_stream(0)
    while True:
        ret, frame = video_stream.read()

        if not ret:
            print("Failed to grab frame from video stream")

        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(16) & 0xFF

        if (key == ESC_KEY):
            cv2.destroyAllWindows()
            return
    return

    last_move = Dir.Stop

    while True:
        img_data = get_capture(0)

        if img_data is None:
            continue
        
        frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(16) & 0xFF

        if (key == ESC_KEY):
            cv2.destroyAllWindows()
            return
        
        if (key == ord('w')):
            last_move = Dir.Forward
            move(0, last_move)
        elif (key == ord('s')):
            last_move = Dir.Backward
            move(0, last_move)
        elif (key == ord('a')):
            last_move = Dir.Left
            move(0, last_move)
        elif (key == ord('d')):
            last_move = Dir.Right
            move(0, last_move)
        elif (key == ord('r')):
            last_move = Dir.Stop
            move(0, last_move)
        else:
            if (last_move != Dir.Stop):
                last_move = Dir.Stop
                move(0, last_move)

        continue
    pass

if __name__ == "__main__":
    main()