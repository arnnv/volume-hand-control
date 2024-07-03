import time
import cv2
import mediapipe as mp
import numpy as np
import HandTracking as ht
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]


cam_width, cam_height = 720, 480


cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

pTime = 0

detector = ht.HandDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 12, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (255, 0, 0), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # FINGER RANGE : 15 to 200
        # VOLUME RANGE : -65.25 to 0

        vol = np.interp(length, [15, 200], [min_vol, max_vol])
        volume.SetMasterVolumeLevel(vol, None)
        print(vol)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
