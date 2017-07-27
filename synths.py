# This program creates a synthetic .avi movie for use with DigiPyRo
# The video shows a ball rolling on a parabolic surface
# The user may change the length of the movie[1], the frame rate of the movie[2], the resolution of the movie[3] 
# the frequency of oscillations[4], the rotation rate of the reference frame[5]
# and control the initial conditions of the roll [6]-[9]

# Import necessary modules
import cv2
import numpy as np
from Tkinter import *
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import leastsq
spinlab = cv2.imread('SpinLabUCLA_BW_strokes.png') # spinlab logo to display in upper right corner of output video

# Ask user for movie name
saveFile = raw_input('Enter a name for the movie (e.g. mySyntheticMovie): ')

# Ask user if they would like a parabolic side view
parView = raw_input('Would you also like a side view? (yes/no): ')
doParView = 'yes' in parView

if doParView:
    saveFilePar = saveFile + 'side.avi'
saveFile += '.avi'

# Define movie details
movLength = 2				# [1] define the desired length of the movie in seconds
fps = 30.0      			# [2] Set this to a low value (10-15) for increased speed or a higher value (30-60) for better results with DigiPyRo
width = 1260				# [3] Width and height in pixels
height = 720    			# [3] Decrease the width and height for increased speed, increase for improved resolution
spinlab = cv2.resize(spinlab,(int(0.2*width),int((0.2*height)/3)), interpolation = cv2.INTER_CUBIC) # resize spinlab logo based on input video dimensions

# Define table values
rpm = 10.0                            	# [4] frequency of oscillations (in RPM). Good values might be 5-15
					# NOTE: A two-dimensional rotating system naturally takes the shape of a parabola.
					# The rotation rate determines the curvature of the parabola, which is why we define the curvature in terms of RPM
rotRate = 10.0				# [5] rotation rate of camera. The two natural frames of reference are with rotRate = 0 and rotRate = rpm


# Set initial conditions
r0 = 1.0                               	# [6] initial radial position of ball. Choose a value between 0 and 1
vr0 = 0.0                              	# [7] initial radial velocity of ball. Good values might be 0-1
phi0 = np.pi/4                         	# [8] initial azimuthal position of ball. Choose a value between 0 and 2*pi
vphi0 = 0.0                             # [9] initial azimuthal velocity of ball. Good values might be 0-1

# Set the amplitude of oscillations to 40% of the smaller dimension
amp = 0
if width > height:
    amp = int(0.5*height)
    ballSize = int(height/30)
else:
    amp = int(0.5*width)
    ballSize = int(width/30)  
 			
omega = (rpm * 2 * np.pi)/60		# calculate angular frequency of oscillations 

# Create movie file
fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
video_writer = cv2.VideoWriter(saveFile, fourcc, fps, (width, height))
if doParView:
    video_writer_par = cv2.VideoWriter(saveFilePar, fourcc, fps, (width, height))

def r(t):
    t1 = (((vr0**2)+((r0**2)*(vphi0**2)))*(np.sin(omega*t)**2))/(omega**2)
    t2 = (1/omega)*(r0*vr0*np.sin(2*omega*t))
    t3 = (r0**2)*(np.cos(omega*t)**2)
    return (t1+t2+t3)**(0.5)

def phi(t):
    y = ((1/omega)*(np.sin(omega*t))*(vr0*np.sin(phi0) + r0*vphi0*np.cos(phi0))) + r0*np.sin(phi0)*np.cos(omega*t)
    x = ((1/omega)*(np.sin(omega*t))*(vr0*np.cos(phi0) - r0*vphi0*np.sin(phi0))) + r0*np.cos(phi0)*np.cos(omega*t)
    return np.arctan2(y,x)

def annotate(img, i): # puts diagnostic text info on each frame
    font = cv2.FONT_HERSHEY_TRIPLEX

    dpro = 'SynthPy'
    dproLoc = (25, 50)
    cv2.putText(img, dpro, dproLoc, font, 1, (255, 105, 180), 1)

    img[25:25+spinlab.shape[0], (width-25)-spinlab.shape[1]:width-25] = spinlab

    timestamp = 'Time: ' + str(round((i/fps),1)) + ' s'
    tLoc = (width - 225, height-25)
    cv2.putText(img, timestamp, tLoc, font, 1, (255, 255, 255), 1)

def parabolaPoints():
    xpoints = np.empty(width)
    ypoints = np.empty(width)
    for i in range(width):
        xpoints[i] = i
        ypoints[i] = int( (height/4) + ((rpm**2)*((i/amp)**2))/(2*9.8))
        nextPoint = np.array([xpoints[i], ypoints[i]])
        try:
            ppoints
        except:
            ppoints = nextPoint
        else:
            ppoints = np.append(ppoints, nextPoint, axis=0)


numFrames = int(movLength * fps)	# calculate number of frames in movie
phi0 *= -1				# correct angle-measuring convention
dtheta = rotRate*(6/fps)                    # rotation of camera for each frame (in degrees)

for i in range(numFrames):
    frame = np.zeros((height,width,3), np.uint8)

    # Outline of circular table
    cv2.circle(frame,(width/2, height/2), int(amp), (255,255,255), 2) 
    
    # Place marker at center of table
    ls = 5 				# line length (in pixels)
    cv2.line(frame, (width/2+ls, height/2), (width/2-ls, height/2), (255,255,255), 1)
    cv2.line(frame, (width/2, height/2+ls), (width/2, height/2-ls), (255,255,255), 1)

    # Calculate new position of ball and draw it
    t = float(i)/fps
    currentPos = ((width/2)+int(amp*r(t)*np.cos(phi(t))), (height/2)+int(amp*r(t)*np.sin(phi(t))))
    cv2.circle(frame, currentPos, ballSize, (255,255,255), -1)
    if rotRate != 0:
        M = cv2.getRotationMatrix2D((int(width/2), int(height/2)), i*dtheta, 1.0)
        frame = cv2.warpAffine(frame, M, (width, height))
    annotate(frame,i)
    frame = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC)
    video_writer.write(frame)

    if doParView:
        framePar = np.zeros((height,width,3), np.uint8)
        parPoints = parabolaPoints()
        cv2.polylines(framePar, parPoints, 0, (255,255,255), 2)
        video_writer_par.write(framePar)

video_writer.release()
video_writer_par.release()
