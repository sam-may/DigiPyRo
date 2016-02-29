import cv2
import numpy as np
import Tkinter as tk

spinlab = cv2.imread('spinlogo.png')

vid = cv2.VideoCapture('/Users/sammay/Desktop/SPINLab/DigiRo/DigiRo-Movies/GoPro_0RPM_JAroll2.mov')
numFrames = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
width = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
#fps = (vid.get(cv2.cv.CV_CAP_PROP_FPS))
fps = 29.97
fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
video_writer = cv2.VideoWriter('/Users/sammay/Desktop/SPINLab/DigiRo/DigiRo-Movies/output.avi', fourcc, fps, (width, height))

physicalRPM = 0
digiRPM = -10
dtheta = digiRPM*(6/fps)
per = 60*(1 / np.abs(float(digiRPM)))

def centerClick(event, x, y, flags, param):
    global center, frame
    clone = frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x,y)
        cv2.circle(frame, (x,y), 4, (255,0,0), -1)
        cv2.imshow('CenterClick', frame)
        frame = clone.copy() # resets to original image so that if the user reselects the center, the old circle will not appear

def centerImg(img, x_c, y_c): # shifts image so that it is centered at (x_c, y_c)
    dx = (width/2) - x_c
    dy = (height/2) - y_c
    shiftMatrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, shiftMatrix, (width, height))

def circumferencePoints(event, x, y, flags, param):
    global npts, center, frame, xpoints, ypoints, r, poly1, poly2
    if event == cv2.EVENT_LBUTTONDOWN:
        if (npts == 0):
            xpoints = np.array([x])
            ypoints = np.array([y])
        else:
            xpoints = np.append(xpoints,x)
            ypoints = np.append(ypoints,y)
        npts+=1
        cv2.circle(frame, (x,y), 3, (0,255,0), -1)
        clone = frame.copy()
        if (len(xpoints) > 2):
            bestfit = calc_center(xpoints, ypoints)
            center = (bestfit[0], bestfit[1])
            r = bestfit[2]
            poly1 = np.array([[0,0],[frame.shape[1],0],[frame.shape[1],frame.shape[0]], [0,frame.shape[0]]])
            poly2 = np.array([[bestfit[0]+r,bestfit[1]]])
            circpts = 100
            for i in range(1,circpts):
                theta =  2*np.pi*(float(i)/circpts)
                nextpt = np.array([[int(bestfit[0]+(r*np.cos(theta))),int(bestfit[1]+(r*np.sin(theta)))]])
                poly2 = np.append(poly2,nextpt,axis=0)
            cv2.circle(frame, center, 4, (255,0,0), -1)
            cv2.circle(frame, center, r, (0,255,0), 1)
        cv2.imshow('CenterClick', frame) 
        frame = clone.copy()

def calc_center(xp, yp):
    n = len(xp)
    circleMatrix = np.matrix([[np.sum(xp**2), np.sum(xp*yp), np.sum(xp)], [np.sum(xp*yp), np.sum(yp**2), np.sum(yp)], [np.sum(xp), np.sum(yp), n]])
    circleVec = np.transpose(np.array([np.sum(xp*((xp**2)+(yp**2))), np.sum(yp*((xp**2)+(yp**2))), np.sum((xp**2)+(yp**2))]))
    ABC = np.transpose(np.dot(np.linalg.inv(circleMatrix), circleVec))
    xc = ABC.item(0)/2
    yc = ABC.item(1)/2
    a = ABC.item(0)
    b = ABC.item(1)
    c = ABC.item(2)
    d = (4*c)+(a**2)+(b**2)
    diam = d**(0.5)
    return np.array([int(xc), int(yc), int(diam/2)])

def annotateImg(img, i):
    font = cv2.FONT_HERSHEY_TRIPLEX

    dpro = 'DigiPyRo'
    dproLoc = (width-200, height-(50+spinlab.shape[0]))
    cv2.putText(img, dpro, dproLoc, font, 1, (255, 255, 255), 1)
    
    img[(height-25)-spinlab.shape[0]:height-25, (width-25)-spinlab.shape[1]:width-25] = spinlab

    perStamp = 'Period (T): ' + str(round(per,1)) + ' s'
    perLoc = (25, height-75)
    cv2.putText(img, perStamp, perLoc, font, 1, (255, 255, 255), 1)
    timestamp = 'Time: ' + str(round(((i/fps)/per),1)) + ' T'
    tLoc = (25, height-25)
    cv2.putText(img, timestamp, tLoc, font, 1, (255, 255, 255), 1)

    prpm = 'Physical Rotation: '
    if (physicalRPM > 0):
        prpm += '+'
    prpm += str(physicalRPM) + 'RPM'
    
    drpm = 'Digital Rotation: '
    if (digiRPM > 0):
        drpm += '+'
    drpm += str(digiRPM) + 'RPM'
    pLoc = (25, 75)
    dLoc = (25, 125)
    cv2.putText(img, prpm, pLoc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, drpm, dLoc, font, 1, (255, 255, 255), 1)

npts = 0

#frames = np.empty(numFrames, dtype = list)

# Open first frame from video, user will click on center
ret, frame = vid.read()
cv2.namedWindow('CenterClick')
cv2.setMouseCallback('CenterClick', circumferencePoints)

cv2.imshow('CenterClick', frame)
cv2.waitKey(0)

for i in range(350):
    ret, frame = vid.read() # read next frame from video

    M = cv2.getRotationMatrix2D(center, i*dtheta, 1.0)
    rotated = cv2.warpAffine(frame, M, (width, height))
    cv2.fillPoly(rotated, np.array([poly1, poly2]), 0)
    cv2.circle(rotated, center, 4, (255,0,0), -1)
    centered = centerImg(rotated, center[0], center[1])
    
    
    centered = cv2.resize(centered,(width,height), interpolation = cv2.INTER_CUBIC)
    annotateImg(centered, i)
    video_writer.write(centered)
    print i


cv2.destroyAllWindows()
vid.release()
video_writer.release()
