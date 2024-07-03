import time
import cv2
import mediapipe as mp
import numpy as np
import HandTracking as ht
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)

    vol_range = volume.GetVolumeRange()
    min_vol, max_vol = vol_range[0], vol_range[1]

    cam_width, cam_height = 720, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, cam_width)
    cap.set(4, cam_height)

    pTime = 0
    detector = ht.HandDetector()

    try:
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            if not success:
                break

            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            vol = 0
            volBar = 0
            volPer = 0

            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(img, (x1, y1), 12, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 12, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                length = math.hypot(x2 - x1, y2 - y1)

                if length < 50:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                vol = np.interp(length, [15, 200], [min_vol, max_vol])
                volBar = np.interp(vol, [min_vol, max_vol], [400, 150])
                volPer = np.interp(vol, [min_vol, max_vol], [0, 100])

                cv2.rectangle(img, (50, int(volBar)), (80, 400), (0, 255, 0), cv2.FILLED)
                volume.SetMasterVolumeLevel(vol, None)

            cv2.rectangle(img, (50, 150), (80, 400), (0, 255, 0), 3)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, f"{int(volPer)}%", (45, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
