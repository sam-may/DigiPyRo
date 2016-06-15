import cv2
import numpy as np
from Tkinter import *
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import leastsq

fps = 30.0
width = 2*1260
height = 2*720
fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
saveFile = '/Users/sammay/Desktop/SPINLab/DigiRo/syntheticMovie.avi'
video_writer = cv2.VideoWriter(saveFile, fourcc, fps, (width, height))

ballSize = 24
yoff = -400
initPos = (width/2, height/2 + yoff)
rpm = 10
omega = (rpm * 2 * np.pi)/60

def harmOsc(t):
    return (width/2, int(height/2 + (yoff*np.cos(omega*t)) + 0.5))

dampCoeff = 0.1
wd = omega*((1-(dampCoeff**2))**(0.5))

def dampOsc(t):
    return (width/2, int(height/2 + (yoff*np.exp(-dampCoeff*omega*t)*np.cos(wd*t)) + 0.5))

def updatePos(oscillatorType, t):
    return oscillatorType(t)

numFrames = 300

for i in range(numFrames):
    frame = np.zeros((height,width, 3), np.uint8)
    #frame = np.random.randint(low=0, high=256, size=(height,width, 3)).astype(np.uint8)
    #frame = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC)

    # Outline of circular table
    cv2.circle(frame,(width/2, height/2), 500, (255,255,255), 2) 
    
    # Place marker at center of table
    ls = 5 # line length (in pixels)
    cv2.line(frame, (width/2+ls, height/2), (width/2-ls, height/2), (255,255,255), 1)
    cv2.line(frame, (width/2, height/2+ls), (width/2, height/2-ls), (255,255,255), 1)

    # Calculate new position of ball and draw it
    currentPos = updatePos(harmOsc, i/fps)
    cv2.circle(frame, currentPos, ballSize, (255,255,255), -1)
    M = cv2.getRotationMatrix2D((width/2, height/2), 45, 1.0)
    frame = cv2.warpAffine(frame, M, (width, height))
    frame = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC)
    video_writer.write(frame)
    print i

video_writer.release()
