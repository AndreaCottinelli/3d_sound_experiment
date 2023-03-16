# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 18:09:34 2022

@author: andre
"""

import cv2
import mediapipe as mp
from time import time
import pyaudio
import numpy as np
import math
import struct
import random
from scipy.ndimage import gaussian_filter

w=640
h=480
n_circ=50
img = np.zeros([h,w],dtype=np.uint8)
img.fill(0)



def createCircle(width, height , rad ):
  wi = random.randint(1, width/2)
  hi = random.randint(1, height/2)
  center = [int(wi), int(hi)]
  radius = np.random.rand()*40+30
  Y, X = np.ogrid[:height, :width]
  dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

  mask = dist_from_center <= radius

  return mask


def addCircle(test_image):
    m = createCircle(width = w, height = h , rad = 30 )
    masked_img = test_image.copy()
    masked_img[m] = np.random.rand()*10
    return masked_img

for i in range(n_circ):
    im = addCircle(test_image = img)
    img = im

maps = np.zeros((h,w,3))

for i in range(3):
    maps[:,:,i] = gaussian_filter(img/10, 15, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

maps=cv2.flip(maps,0)
#cv2.imshow("Image", maps)    



TT = time()
freq = 200
newfreq = 200
phase = 0
FS = 44100  #  frames per second, samples per second or sample rate
vol=0.5

def generate_sound(type, frequency, volume, duration):

    outbuf = np.random.normal(loc=0, scale=1, size=int(float(duration / 1000.0)*FS))

    if type == "sine":
        dur = int(FS * float(duration / 1000.0))
        theta = 0.0
        incr_theta = frequency * 2 * math.pi / FS # frequency increment normalized for sample rate
        for i in range(dur):
            outbuf[i] = volume * math.sin(theta)
            theta += incr_theta

    p = pyaudio.PyAudio()
    stream = p.open(formato=pyaudio.paFloat32, 
                    channels=2, 
                    rate=FS, 
                    output=True,
                    output_device_index=3)
    
    data = b''.join(struct.pack('f', samp) for samp in outbuf) # must pack the binary data
    stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
def callback(in_data, frame_count, time_info, status):
    global TT,phase,freq,newfreq, vol
    if newfreq != freq:
        phase = 2*np.pi*TT*(freq-newfreq)+phase
        freq=newfreq
    left = (np.sin(phase+2*np.pi*freq*(TT+np.arange(frame_count)/float(FS))))
    data = np.zeros((left.shape[0]*2,),np.float32)
    data[::2] = left
    data[1::2] = left
    data=data*vol
    TT+=frame_count/float(FS)
    return (data, pyaudio.paContinue)    


#% MAIN
    
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw= mp.solutions.drawing_utils

pTime = 0
cTime = 0

#alph=list(string.ascii_uppercase)

p = pyaudio.PyAudio()

stream = p.open(formato=pyaudio.paFloat32,
                channels=2,
                rate=FS,
                output=True,
                stream_callback=callback,
                output_device_index=3)

stream.start_stream()
start=time()

while True:
    now = time()
    success, img = cap.read()
    fin_img=maps
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    
    h, w, c = img.shape
    wfrac = 80
    hfrac = 60
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for idd, lm in enumerate(handLms.landmark):
                
                
                cx, cy = int(lm.x*w), int(lm.y*h)
                if idd == 8:
    
                    cv2.circle(img,(cx,cy), 10, (255,0,255), cv2.FILLED)
                    binx=(cx) // (w/(wfrac)) 
                    biny=(cy) // (h/(hfrac)) 
                    
                    #print(idd, cx, cy)
                    
                    #freqgen = (hfrac - biny) / hfrac *40 + 220 - 40/2 
                    #vol     = np.min([0.999, (binx) / wfrac]) 
                    
                    cy = np.min([cy, h-1])
                    cy = np.max([cy, 1])
                    cx = np.min([cx, w-1])
                    cx = np.max([cx, 1])
                    r = cx
                    c = cy
                    freqgen = maps[r, c, 0] *40 +220 -40/2 
                    print(cx,cy)
                    cv2.putText(maps, "freq: " + str(int(freq)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
                    cv2.putText(img, "vol: " + str(vol), (10,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
                    fin_img = cv2.rectangle(maps, (cx-1,cy-1), (cx, cy), (0,0,255), -1)
                    
                    if now-start > 1./24: #aggiorno la frequenza solo 24 volte al secondo!
                        newfreq= freqgen 
                        #print(newfreq)
                        start=now
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # draw connections between landmarks
            
    #cTime = time()
    #fps= 1/(cTime-pTime)
    #pTime=cTime
    
    
    
    #wf=int(w/wfrac)
    #hf=int(h/hfrac)
    #color = (255, 0, 0)
    #thickness = 1
    
    #for ww in range(wfrac):
    #    for hh in range(hfrac):
    # 
    #         start_point = (ww*wf, hh*hf)
    #        end_point = ((ww+1)*wf, (hh+1)*hf)
    #        #celltxt= alph[hh*wfrac+ww]
    #        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    #        #cv2.putText(img, str(ww)+ "," + str(hh), (int((ww+0.5)*wf), int((hh+0.5)*hf)), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
    
    
    #cv2.putText(fin_img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    #cv2.putText(fin_img, quad1 + " " + quad2, (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    #added_img= cv2.addWeighted(img, 0.4, fin_img, 0.1,0)
    cv2.imshow("Image", fin_img)
    cv2.waitKey(1)
    

#%%
cap = cv2.VideoCapture(0)
success, img = cap.read()
img = cv2.flip(img, 1)

cv2.imshow("Image", fin_img)

