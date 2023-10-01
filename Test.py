
import pyttsx3
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

offsets = 20
imagesize = 300
counter = 0
folder = "DATA/3"
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')
lable = ['1', '2', '3', '4','5','6','7','8','9','10',"Sad"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            predection, index = classifier.getPrediction(imgWhite, draw = False)
            print( index+1)
        else:
            k = imagesize / h
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imagesize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imagesize - hCal /1))
            imgWhite[hGap:hCal + hGap, :] = imgResize
            predection, index = classifier.getPrediction(imgWhite, draw = False)
        cv2.putText(imgOutput, lable[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x-offsets, y-offsets), (x+w+offsets,y+h+offsets), (255, 0, 255), 4)
        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("Image White", imgWhite)

        s = lable[index]
        speak(s)
        print(s)



    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

