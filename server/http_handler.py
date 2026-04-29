import requests
from time import sleep
from secrets import IPs

import numpy as np
import cv2


def get_capture_url(robot_id:int):
    if (0 <= robot_id < len(IPs)):
        return f"http://{IPs[robot_id]}/capture"
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


def main():
    ESC_KEY = 27

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

        continue
    pass

if __name__ == "__main__":
    main()