# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:44:46 2020

@author: SUDARSHAN
"""


import cv2

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#creating the video cature object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # converting frames colour to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detecting the faces
    detections = cascade_classifier.detectMultiScale(gray,1.1,5)
    #drawing rectange over the detected faces
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)



    # Display the resulting frame
    cv2.imshow('frame',frame)
    #loop terminating condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()