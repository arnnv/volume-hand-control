import time
import cv2
import mediapipe as mp
import numpy as np
import HandTracking as ht

cam_width, cam_height = 1280, 720


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
        print(lmList[4], lmList[8])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
