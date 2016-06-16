# This program creates a synthetic .avi movie for use with DigiPyRo
# The video shows a ball rolling on a parabolic surface
# The user may change the length of the movie[1], the frame rate of the movie[2], the resolution of the movie[3] 
# the frequency of oscillations[4], the size of the ball[5], and choose to add/remove frictional effects[6]

# Import necessary modules
import cv2
import numpy as np
from Tkinter import *
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import leastsq

# Ask user for movie name
saveFile = raw_input('Enter a name for the movie (e.g. mySyntheticMovie): ')
saveFile += '.avi'

# Define movie details
movLength = 10				# [1] define the desired length of the movie in seconds
fps = 30.0      			# [2] Set this to a low value (10-15) for increased speed or a higher value (30-60) for better results with DigiPyRo
width = 1260				# [3] Width and height in pixels
height = 720    			# [3] Decrease the width and height for increased speed, increase for improved resolution

# Define ball and table values
rpm = 10                                # [4] frequency of oscillations (in RPM). Good values might be 5-15
ballSize = 24				# [5] radius of ball in pixels. Good values might be 15-30
friction = False			# [6] set to "False" for no friction (harmonic oscillator) or "True" to add friction (damped harmonic oscillator)
dampCoeff = 0.1				# [6] coefficient of friction (applicable only if "friction = True"). Good values might be 0.1-0.2


# Set the amplitude of oscillations to 40% of the smaller dimension
amp = 0
if width > height:
    amp = int(0.4*height)
else:
    amp = int(0.4*width)  
 				
initPos = (width/2, height/2 - amp) 	# initial xy-coordinates of ball
omega = (rpm * 2 * np.pi)/60		# calculate angular frequency of oscillations 



# Create movie file
fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
video_writer = cv2.VideoWriter(saveFile, fourcc, fps, (width, height))

def harmOsc(t):
    return (width/2, int(height/2 + (amp*np.cos(omega*t)) + 0.5))

wd = omega*((1-(dampCoeff**2))**(0.5))	# calculate frequency of damped oscillations

def dampOsc(t):
    return (width/2, int(height/2 + (amp*np.exp(-dampCoeff*omega*t)*np.cos(wd*t)) + 0.5))

def updatePos(oscillatorType, t):
    return oscillatorType(t)

numFrames = int(movLength * fps)	# calculate number of frames in movie

oscType = 0
if friction:
    oscType = dampOsc
else:
    oscType = harmOsc


for i in range(numFrames):
    frame = np.zeros((height,width,3), np.uint8)

    # Outline of circular table
    cv2.circle(frame,(width/2, height/2), int(1.1*amp), (255,255,255), 2) 
    
    # Place marker at center of table
    ls = 5 				# line length (in pixels)
    cv2.line(frame, (width/2+ls, height/2), (width/2-ls, height/2), (255,255,255), 1)
    cv2.line(frame, (width/2, height/2+ls), (width/2, height/2-ls), (255,255,255), 1)

    # Calculate new position of ball and draw it
    currentPos = updatePos(oscType, i/fps)
    cv2.circle(frame, currentPos, ballSize, (255,255,255), -1)
    M = cv2.getRotationMatrix2D((width/2, height/2), 45, 1.0)
    frame = cv2.warpAffine(frame, M, (width, height))
    frame = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC)
    video_writer.write(frame)

video_writer.release()
