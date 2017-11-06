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


# Define movie details
movLength = 5                           # [1] define the desired length of the movie in seconds
fps = 30.0                              # [2] Set this to a low value (10-15) for increased speed or a higher value (30-60) for better results with DigiPyRo
width = 1260                            # [3] Width and height in pixels
height = 720                            # [3] Decrease the width and height for increased speed, increase for improved resolution
spinlab = cv2.resize(spinlab,(int(0.2*width),int((0.2*height)/3)), interpolation = cv2.INTER_CUBIC)


# Define table values
rpm = 10.0                              # [4] frequency of oscillations (in RPM). Good values might be 5-15
                                        # NOTE: A two-dimensional rotating system naturally takes the shape of a parabola.
                                        # The rotation rate determines the curvature of the parabola, which is why we define the curvature in terms of RPM
rotRate = 0.0                          # [5] rotation rate of camera. The two natural frames of reference are with rotRate = 0 and rotRate = rpm


# Set initial conditions
r0 = 1.0                                # [6] initial radial position of ball. Choose a value between 0 and 1
vr0 = 0.0                               # [7] initial radial velocity of ball. Good values might be 0-1
phi0 = np.pi/4                          # [8] initial azimuthal position of ball. Choose a value between 0 and 2*pi
vphi0 = 0.0                             # [9] initial azimuthal velocity of ball. Good values might be 0-1


# Ask user for movie name
saveFile = raw_input('Enter a name for the movie (e.g. mySyntheticMovie): ')

# Ask user if they would like a parabolic side view
parView = raw_input('Would you also like a side view? (yes/no): ')
doParView = 'yes' in parView

if doParView:
    saveFilePar = saveFile + 'side.avi'
saveFile += '.avi'

if doParView:
    parViewFrac = 0.3
    borderHeight = 50
    lineWidth = 10
    parHeight = int(height*(parViewFrac+1))
    fullHeight = parHeight + borderHeight
else:
    fullHeight = height


# No longer asking user if they care about reference frame of parabolic side view
doParViewRot = (rotRate != 0)

# Ask user if they care about reference frame of parabolic side view
#doParViewRot = False
#if doParView and rotRate!=0:
#    parViewRot = raw_input('Would you like the parabolic side view in the rotating frame? (yes/no): ')
#    doParViewRot = 'yes' in parViewRot

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
video_writer = cv2.VideoWriter(saveFile, fourcc, fps, (width, fullHeight))
#if doParView:
#    video_writer_par = cv2.VideoWriter(saveFilePar, fourcc, fps, (width, height))

def r(t):
    t1 = (((vr0**2)+((r0**2)*(vphi0**2)))*(np.sin(omega*t)**2))/(omega**2)
    t2 = (1/omega)*(r0*vr0*np.sin(2*omega*t))
    t3 = (r0**2)*(np.cos(omega*t)**2)
    return (t1+t2+t3)**(0.5)

def phi(t):
    y = ((1/omega)*(np.sin(omega*t))*(vr0*np.sin(phi0) + r0*vphi0*np.cos(phi0))) + r0*np.sin(phi0)*np.cos(omega*t)
    x = ((1/omega)*(np.sin(omega*t))*(vr0*np.cos(phi0) - r0*vphi0*np.sin(phi0))) + r0*np.cos(phi0)*np.cos(omega*t)
    return np.arctan2(y,x)

def annotate(img, i, rotatingView): # puts diagnostic text info on each frame
    font = cv2.FONT_HERSHEY_TRIPLEX

    dpro = 'SynthPy'
    dproLoc = (25, 50)
    cv2.putText(img, dpro, dproLoc, font, 1, (255,255,255), 1)

    topView = 'Top View'
    topViewLoc = (25, 90)
    cv2.putText(img, topView, topViewLoc, font, 1, (255,255,255), 1)

    if rotatingView:
        rotView = 'Rotating View'
        rotViewLoc = (25, 130)
        cv2.putText(img, rotView, rotViewLoc, font, 1, (55, 255, 90), 1)
    else:
        rotView = 'Inertial View'
        rotViewLoc = (25, 130)
        cv2.putText(img, rotView, rotViewLoc, font, 1, (255, 105, 180), 1)

    img[25:25+spinlab.shape[0], (width-25)-spinlab.shape[1]:width-25] = spinlab

    timestamp = 'Time: ' + str(round((i/fps),1)) + ' s'
    tLoc = (width - 225, height-25)
    cv2.putText(img, timestamp, tLoc, font, 1, (255, 255, 255), 1)

    rad = 'Radius: R = 1 m'
    radLoc = (width -325, height-65)
    cv2.putText(img, rad, radLoc, font, 1, (255, 255, 255), 1)

def parabolaPoints():
    xpoints = np.empty(width)
    ypoints = np.empty(width)
    metersToPixels = float(amp)/2
    for i in range(width):
        if i < (width/2 - int(amp)) or i > (width/2 + int(amp)):
            continue
        xpoints[i] = i
        #ypoints[i] = int( ((0.75)*float(fullHeight-height)) - ((omega**2)*((float(i-(width/2))/float(amp))*)/(2*9.8))
        #ypoints[i] = int( ((0.75)*float(fullHeight-height)) - ((rpm**2)*((float(i-(width/2))/float(amp))**2))/(2*9.8))
        ypoints[i] = int( ((0.75)*float(fullHeight-height)) - ((omega**2)*((float(i-(width/2))/float(amp))**2)*((metersToPixels)**2))/(2*9.8*metersToPixels))
        nextPoint = np.array([xpoints[i], ypoints[i]])
        try:
            ppoints
        except:
            ppoints = nextPoint
        else:
            ppoints = np.append(ppoints, nextPoint, axis=0)
    return ppoints

def parabola(x):
    metersToPixels = float(amp)/2
    return int( ((0.75)*float(fullHeight-height)) - ((omega**2)*((float(x-(width/2))/float(amp))**2)*((metersToPixels)**2))/(2*9.8*metersToPixels))


def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
   #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

   #difference and absolute difference between points
   #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

   #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

   #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

   #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

   #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer


def dottedLine(frame, xi, yi, xf, yf, c1, c2, c3, thickness, segmentLength):
    it = createLineIterator((xi,yi), (xf, yf), frame[:,:,0])
    totLength = (xf-xi)**2 + (yf-yi)**2
    nLines = int(totLength/segmentLength)
    for i in range(nLines):
        if i%2 == 0:
            continue
        #if (i)*segmentLength >= it.shape[0]:
        #    continue
        try:
            cv2.line(frame, (it[i*segmentLength,0], it[i*segmentLength,1]), (it[(i+1)*segmentLength, 0], it[(i+1)*segmentLength, 1]), (c1, c2, c3), thickness)
        except:
            continue

def rotatedDottedLine(theta, frame, xi, yi, xf, yf, c1, c2, c3, thickness, segmentLength):
    centerX = int((xf+xi)/2)
    centerY = int((yf+yi)/2)
    lineRadius = int((((xf-xi)**2 + (yf-yi)**2)**(0.5))/2)
    nXi = int(-lineRadius*np.cos(theta)) + centerX
    nXf = int(lineRadius*np.cos(theta)) + centerX
    nYi = int(lineRadius*np.sin(-theta)) + centerY
    nYf = int(lineRadius*np.sin(theta)) + centerY
    dottedLine(frame, nXi, nYi, nXf, nYf, c1, c2, c3, thickness, segmentLength)

def annotateSideview(img):
    font = cv2.FONT_HERSHEY_TRIPLEX

    dpro = 'Side-View'
    dproLoc = (25, height+borderHeight+25)
    cv2.putText(img, dpro, dproLoc, font, 1, (255,255,255), 1)

    surf = str(rpm) + ' RPM Parabolic Surface'
    surfLoc = (width-500, height+borderHeight+25)
    cv2.putText(img, surf, surfLoc, font, 1, (255,255,255), 1)

    maxDef = ((omega**2))/(2*9.8)
    defl = 'Max. Deflection: h = '+ str(round(maxDef,1)) + ' m'
    deflLoc = (width-500, height+borderHeight+65)
    cv2.putText(img, defl, deflLoc, font, 1, (255,255,255), 1)

#def parHeight(x, phi0):
    

#def annotateSideview(img, i): # puts diagnostic text info on each frame
#    font = cv2.FONT_HERSHEY_TRIPLEX
#
#    dpro = 'SynthPy: Parabolic Side-View'
#    dproLoc = (25, 50)
#    cv2.putText(img, dpro, dproLoc, font, 1, (255, 105, 180), 1)
#    surf = str(rpm) + ' RPM Parabolic Surface'
#    surfLoc = (20, 85)
#    cv2.putText(img, surf, surfLoc, font, 1, (255, 105, 180), 1)
#
#    img[25:25+spinlab.shape[0], (width-25)-spinlab.shape[1]:width-25] = spinlab
#
#    timestamp = 'Time: ' + str(round((i/fps),1)) + ' s'
#    tLoc = (width - 225, height-25)
#    cv2.putText(img, timestamp, tLoc, font, 1, (255, 255, 255), 1)

numFrames = int(movLength * fps)	# calculate number of frames in movie
phi0 *= -1				# correct angle-measuring convention
dtheta = rotRate*(6/fps)                    # rotation of camera for each frame (in degrees)

parPoints = np.int32(parabolaPoints()) # cv2 only accepts int32 arrays for polypoints :-|
parPoints = parPoints.reshape((-1,1,2))

# create map ("dictionary") that allows me to look up the corresponding y-position of any x-point on parabola
parDict = {}
for i in range(parPoints.shape[0]):
    parDict[str(parPoints[i,0,0])] = parPoints[i,0,1] 

for i in range(numFrames):
    frame = np.zeros((height,width,3), np.uint8)

    # Outline of circular table
    cv2.circle(frame,(width/2, height/2), int(amp), (255,255,255), 2) 
    
    # Place marker at center of table
    ls = 5 				# line length (in pixels)
    cv2.line(frame, (width/2+ls, height/2), (width/2-ls, height/2), (255,255,255), 1)
    cv2.line(frame, (width/2, height/2+ls), (width/2, height/2-ls), (255,255,255), 1)

    dottedLine(frame, int(width/2)-int(amp), int(height/2), int(width/2)+int(amp), int(height/2), 255, 105, 180, 2, 10) # for rotating
    if rotRate != 0:
        M = cv2.getRotationMatrix2D((int(width/2), int(height/2)), -i*dtheta*2, 1.0)
        frame = cv2.warpAffine(frame, M, (width, height))

    # Calculate new position of ball and draw it
    t = float(i)/fps
    currentPos = ((width/2)+int(amp*r(t)*np.cos(phi(t))), (height/2)+int(amp*r(t)*np.sin(phi(t))))
    cv2.circle(frame, currentPos, ballSize, (255,255,255), -1)
    if rotRate != 0:
        M = cv2.getRotationMatrix2D((int(width/2), int(height/2)), i*dtheta, 1.0)
        #dottedLine(frame, int(width/2)-int(amp), int(height/2), int(width/2)+int(amp), int(height/2), 255, 105, 180, 2, 10) # for inertial
        frame = cv2.warpAffine(frame, M, (width, height))
        #rotatedDottedLine(-i*dtheta, frame, int(width/2)-int(amp), int(height/2), int(width/2)+int(amp), int(height/2), 255, 105, 180, 2, 10) # for inertial
    annotate(frame,i, rotRate != 0)
    #if rotRate != 0:
        #dottedLine(frame, int(width/2)-int(amp), int(height/2), int(width/2)+int(amp), int(height/2), 55, 255, 90, 2, 10) # for rotating
    if rotRate == 0:
        dottedLine(frame, int(width/2)-int(amp), int(height/2), int(width/2)+int(amp), int(height/2), 255, 105, 180, 2, 10) # for inertial
   #cv2.line(frame, (int(width/2)-int(amp), int(height/2)), (int(width/2)+int(amp), int(height/2)), (255, 105, 180), 2)
    frame = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC)
    if not doParView:
        video_writer.write(frame)
    else:
        fullFrame = np.zeros((fullHeight,width,3), np.uint8)

        # Create parView pad
        framePar = np.zeros((fullHeight-(height+borderHeight),width,3), np.uint8)
        if rotRate != 0:
            cv2.polylines(framePar, [parPoints], 0, (55, 255, 90), 2)
        else:
            cv2.polylines(framePar, [parPoints], 0, (255, 105, 180), 2)

        dottedLine(framePar, width/2-int(amp), 30, width/2-int(amp), framePar.shape[0], 255, 255, 255, 2, 10)
        dottedLine(framePar, width/2+int(amp), 30, width/2+int(amp), framePar.shape[0], 255, 255, 255, 2, 10)
        #cv2.line(framePar, (width/2-int(amp), 30), (width/2-int(amp), framePar.shape[0]), (255,255,255), 2)
        #cv2.line(framePar, (width/2+int(amp), 30), (width/2+int(amp), framePar.shape[0]), (255,255,255), 2)
        if doParViewRot:
            xpos = (width/2) + int(amp*r(t)*np.cos(phi(t)-((i*dtheta)*(3.14159/180))))
        else:
            xpos = (width/2) + int(amp*r(t)*np.cos(phi(t)))
        currentPos = (xpos, parabola(xpos)-ballSize)
        cv2.circle(framePar, currentPos, ballSize, (255,255,255), -1)
        #annotateSideview(framePar, i)
        #video_writer_par.write(framePar)

        fullFrame[0:height,0:width] = frame

        fullFrame[height+lineWidth:height+(borderHeight-lineWidth),0:width] = 255*np.ones((borderHeight-(2*lineWidth), width, 3), np.uint8)
        fullFrame[height+borderHeight:fullHeight,0:width] = framePar


        dottedLine(fullFrame, width/2-int(amp), height/2, width/2-int(amp), fullFrame.shape[0], 255, 255, 255, 2, 10) 
        dottedLine(fullFrame, width/2+int(amp), height/2, width/2+int(amp), fullFrame.shape[0], 255, 255, 255, 2, 10)
        #cv2.line(fullFrame, (width/2-int(amp), height/2), (width/2-int(amp), fullFrame.shape[0]), (255,255,255), 2)
        #cv2.line(fullFrame, (width/2+int(amp), height/2), (width/2+int(amp), fullFrame.shape[0]), (255,255,255), 2)
        annotateSideview(fullFrame)
        video_writer.write(fullFrame)

    #if doParView:
    #    framePar = np.zeros((height,width,3), np.uint8)
    #    cv2.polylines(framePar, [parPoints], 0, (255,255,255), 2)
    #    currentPos = ((width/2) + int(amp*r(t)*np.cos(phi(t))), parPoints[i,0,1])
    #    cv2.circle(framePar, currentPos, ballSize, (255,255,255), -1)
    #    annotateSideview(framePar, i)
    #    video_writer_par.write(framePar)



video_writer.release()
