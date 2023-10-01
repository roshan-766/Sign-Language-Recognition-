
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
#--------------------------------------------------------------------------------------------------------
nlable = input('Enter The Sign Name :')
os.mkdir("DATA/"+ nlable)

#---------------------------------------------------------------------------------------------------------
offsets = 20
imagesize = 300
counter = 0
folder = "DATA/"+nlable

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imagesize, imagesize, 3), np.uint8)*255
        imgCrop = img[y-offsets:y+h+offsets, x-offsets:x+w+offsets]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imagesize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imagesize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imagesize-wCal/1))
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imagesize / h
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imagesize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imagesize - hCal /1))
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)


