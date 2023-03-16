# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 20:30:40 2022

@author: Andrea
"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print("[INFO] handmarks: {}".format(results.multi_hand_landmarks))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for  lm in hand_landmarks.landmark:
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#%%
import cv2
import numpy as np
import time

cam = cv2.VideoCapture(1)
time.sleep(2)

while True:
    ret,frame = cam.read()
    #cv2.imshow('webcam', frame)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    #if cv2.waitKey(1)&0xFF == ord('q'):
        #break

cam.release()
cv2.destroyAllWindows()

#%%
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

prev_time = 0
cur_time = 0


while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    results = hands.process(imgRGB)
    # print("[INFO] handmarks: {}".format(results.multi_hand_landmarks))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for index, lm in enumerate(hand_landmarks.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cur_time = time.time()
    fps = 1/(cur_time-prev_time)
    prev_time = cur_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    img = cv2.imread('Image')
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break