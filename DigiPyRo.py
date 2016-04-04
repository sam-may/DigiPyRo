import cv2
import numpy as np
from Tkinter import *
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import leastsq

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

def locate(event, x, y, flags, param):
    global frame, particleStart, particleEnd, particleCenter, particleRadius
    clone = frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        particleStart = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        particleEnd = (x,y)
        particleCenter = ((particleEnd[0] + particleStart[0])/2, (particleEnd[1] + particleStart[1])/2)
        d2 = ((particleEnd[0] - particleStart[0])**2) + ((particleEnd[1] - particleStart[1])**2)
        particleRadius = (d2**(0.5))/2
        cv2.circle(frame, particleCenter, int(particleRadius+0.5), (255,0,0), 1)
        cv2.imshow('Locate Ball', frame)
        frame = clone.copy() # resets to original image

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
 
def removePoint(orig):
    global npts, center, frame, xpoints, ypoints, r, poly1, poly2
    if npts == 0:
        return

    else:
        npts -= 1
        if npts == 0:
            xpoints = np.empty(0)
            ypoints = np.empty(0)
        elif npts == 1:
            xpoints = np.array([xpoints[0]])
            ypoints = np.array([ypoints[0]])
        else:
            xpoints = xpoints[0:npts]
            ypoints = ypoints[0:npts]

    frame = orig.copy()
    for i in range(len(xpoints)):
        cv2.circle(frame, (xpoints[i], ypoints[i]), 3, (0,255,0), -1)
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
    dproLoc = (25, 50)
    cv2.putText(img, dpro, dproLoc, font, 1, (255, 255, 255), 1)
    
    #img[(height-25)-spinlab.shape[0]:height-25, (width-25)-spinlab.shape[1]:width-25] = spinlab
    img[25:25+spinlab.shape[0], (width-25)-spinlab.shape[1]:width-25] = spinlab

    #perStamp = 'Period (T): ' + str(round(per,1)) + ' s'
    #perLoc = (25, height-75)
    #cv2.putText(img, perStamp, perLoc, font, 1, (255, 255, 255), 1)
    #timestamp = 'Time: ' + str(round(((i/fps)/per),1)) + ' T'
    timestamp = 'Time: ' + str(round((i/fps),1)) + ' s'
    tLoc = (width - 225, height-25)
    cv2.putText(img, timestamp, tLoc, font, 1, (255, 255, 255), 1)

    prpm = 'Physical Rotation: '
    if (physicalRPM > 0):
        prpm += '+'
    prpm += str(physicalRPM) + 'RPM'
    
    drpm = 'Additional Digital Rotation: '
    if (digiRPM > 0):
        drpm += '+'
    drpm += str(digiRPM) + 'RPM'
    pLoc = (25, height - 25)
    dLoc = (25, height - 75)
    cv2.putText(img, prpm, pLoc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, drpm, dLoc, font, 1, (255, 255, 255), 1)

def instructsCenter(img):
    font = cv2.FONT_HERSHEY_PLAIN
    line1 = 'Click on 3 or more points along the border of the circle'
    line1Loc = (25, 50)
    line2 = 'around which the movie will be rotated.'
    line2Loc = (25, 75)
    line3 = 'Press the BACKSPACE or DELETE button to undo a point.'
    line3Loc = (25,100)
    line4 = 'Press ENTER when done.'
    line4Loc = (25,125) 
    
    cv2.putText(img, line1, line1Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line2, line2Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line3, line3Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line4, line4Loc, font, 1, (255, 255, 255), 1)

def instructsBall(img):
    font = cv2.FONT_HERSHEY_PLAIN
    line1 = 'Click and drag to create a circle around the ball.'
    line1Loc = (25, 50)
    line2 = 'The more accurately the initial location and size of the ball'
    line2Loc = (25, 75)
    line3 = 'are matched, the better the tracking results will be.'
    line3Loc = (25, 100)
    line4 = 'Press ENTER when done.'
    line4Loc = (25, 125)

    cv2.putText(img, line1, line1Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line2, line2Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line3, line3Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line4, line4Loc, font, 1, (255, 255, 255), 1)

def errFuncPolar(params, data):
    modelR = np.abs(params[0]*np.exp(-data[0]*params[3]*params[1])*np.cos((params[3]*data[0]*((1-(params[1]**2))**(0.5))) - params[2]))
    modelTheta = createModelTheta(data[0], params, data[2][0], data[3])
    model = np.append(modelR, modelR*modelTheta)
    datas = np.append(data[1], data[1]*data[2])
    return model - datas

def fitDataPolar(data):
    result = sp.optimize.leastsq(errFuncPolar, np.array([100, 0.1, 0, 1]), args=(data), full_output=1)
    return result[0]

def createModelR(bestfit, t, omega):
    return np.abs(bestfit[0]*np.exp(-t*omega*bestfit[1])*np.cos((omega*t*((1-(bestfit[1]**2))**(0.5)) - bestfit[2])))

def createModelTheta(t, bestfit, thetai, rot):
    wd = bestfit[3] * ((1 - (bestfit[1])**2)**(0.5))
    period = (2*np.pi)/wd
    phi = bestfit[2]
    angularRot = rot*(-(2*np.pi)/60)
    theta = np.ones(len(t))*thetai
    for i in range(len(t)):
        phase = (wd*t[i])-phi
        while phase > 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi

        if phase < (np.pi/2) or phase > ((3*np.pi)/2):
           theta[i] = thetai
        elif phase > (np.pi/2) and phase < ((3*np.pi)/2):
           theta[i] = thetai + np.pi
        theta[i] += t[i]*angularRot
        
        while theta[i] > 2*np.pi:
           theta[i] -= 2*np.pi
        while theta[i] < 0:
           theta[i] += 2*np.pi
     
    return theta

def start():
    vid = cv2.VideoCapture(filenameVar.get())

    global width, height, numFrames, fps, fourcc, video_writer, spinlab, npts
    npts = 0
    spinlab = cv2.imread('/Users/sammay/Desktop/SPINLab/DigiRo/spinlogo.png')
    #spinlab = cv2.resize(spinlab,spinlab.shape, interpolation = cv2.INTER_CUBIC)
    #numFrames = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = fpsVar.get()
    fileName = savefileVar.get()
    fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
    video_writer = cv2.VideoWriter(fileName+'.avi', fourcc, fps, (width, height))

    global physicalRPM, digiRPM, dtheta, per
    physicalRPM = physRPMVar.get()
    digiRPM = digiRPMVar.get()
    dtheta = digiRPM*(6/fps)
    per = 60*(1 / np.abs(float(digiRPM)))

    startFrame = fps*startTimeVar.get()
    numFrames = int(fps*(endTimeVar.get() - startTimeVar.get()))
    trackBall = trackVar.get()

    # Close GUI window so rest of program can run
    root.destroy()

    global center, frame, xpoints, ypoints, r, poly1, poly2

    # Open first frame from video, user will click on center
    vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, startFrame)
    ret, frame = vid.read()
    cv2.namedWindow('CenterClick')
    cv2.setMouseCallback('CenterClick', circumferencePoints)

    instructsCenter(frame)
    orig = frame.copy()
    while(1):
        cv2.imshow('CenterClick', frame)
        k = cv2.waitKey(0)
        if k == 13: # user presses ENTER
            break
        elif k == 127: # user presses BACKSPACE/DELETE
            removePoint(orig)

    cv2.destroyWindow('CenterClick')

    # Select initial position of ball
    if trackBall:
        vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, startFrame)
        ret, frame = vid.read()
        cv2.namedWindow('Locate Ball')
        cv2.setMouseCallback('Locate Ball', locate)

        instructsBall(frame)
        cv2.imshow('Locate Ball', frame)
        cv2.waitKey(0)
        cv2.destroyWindow('Locate Ball')

    vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, startFrame)
    t = np.empty(numFrames)
    ballX = np.empty(numFrames)
    ballY = np.empty(numFrames)
    ballPts = 0 #will identify the number of times that Hough Circle transform identifies the ball
    lastLoc = particleCenter
    thresh = 50
    for i in range(numFrames):
        ret, frame = vid.read() # read next frame from video

        M = cv2.getRotationMatrix2D(center, i*dtheta, 1.0)
        rotated = cv2.warpAffine(frame, M, (width, height))
        cv2.fillPoly(rotated, np.array([poly1, poly2]), 0)
        cv2.circle(rotated, center, 4, (255,0,0), -1)
        centered = centerImg(rotated, center[0], center[1])
    
    
        centered = cv2.resize(centered,(width,height), interpolation = cv2.INTER_CUBIC)
        
        if trackBall:
            gray = cv2.cvtColor(centered, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray,5)
            ballLoc = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius = int(particleRadius * 0.7), maxRadius = int(particleRadius * 1.3))
            if type(ballLoc) != NoneType :
                for j in ballLoc[0,:]:
                    if (np.abs(j[0] - lastLoc[0]) < thresh) and (np.abs(j[1] - lastLoc[1]) < thresh):    
                        cv2.circle(centered, (j[0],j[1]), j[2], (0,255,0),1)
                        cv2.circle(centered, (j[0],j[1]), 2, (0,0,255), -1)
                        ballX[ballPts] = j[0]
                        ballY[ballPts] = j[1]
	                t[ballPts] = i/fps
                        lastLoc = np.array([j[0],j[1]])      
                        ballPts += 1
                        break
           
            for k in range(ballPts-1):
                cv2.circle(centered, (int(ballX[k]), int(ballY[k])), 1, (255,0,0), -1)    	


        annotateImg(centered, i)
        video_writer.write(centered)

    if trackBall:
        ballX = ballX[0:ballPts]
        ballY = ballY[0:ballPts]
        t = t[0:ballPts]
        ballX -= center[0]
        ballY -= center[1]
    
        ballR = ((ballX**2)+(ballY**2))**(0.5)
        ballTheta = np.arctan2(ballY, ballX)
        for i in range(len(ballTheta)):
            if ballTheta[i] < 0:
                ballTheta[i] += 2*np.pi

        dataFile = open(fileName+'_data.txt', 'w')
        dataFile.write('x y r theta\n')
        for i in range(len(ballX)):
            dataFile.write(str(ballX[i])+' '+str(ballY[i])+' '+str(ballR[i])+' '+str(ballTheta[i])+'\n')
        dataFile.close()

        omega = (np.pi)/3
        dataFitPolar = fitDataPolar(np.array([t, ballR, ballTheta, digiRPM + physicalRPM]))
        print dataFitPolar
        modelR = createModelR(dataFitPolar, t, omega)
        modelTheta = createModelTheta(t, dataFitPolar, ballTheta[0], digiRPM + physicalRPM)
        plt.figure(2)
        plt.subplot(211)
        plt.plot(t, ballR, 'r1')
        plt.plot(t, modelR, 'b')
        plt.xlabel(r"$t$ (s)")
        plt.ylabel(r"$r$")

        plt.subplot(212)
        plt.plot(t, ballTheta, 'r1')
        plt.plot(t, modelTheta, 'b')
        plt.xlabel(r"$t$ (s)")
        plt.ylabel(r"$\theta$")
        plt.savefig('/Users/sammay/Desktop/SPINLab/DigiRo/'+fileName+'_polar.pdf', format = 'pdf', dpi = 1200)

        modelX = modelR*np.cos(modelTheta)
        modelY = modelR*np.sin(modelTheta)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, ballX, 'r1')
        plt.plot(t, modelX, 'k')
        plt.xlabel(r"$t$ (s)")
        plt.ylabel(r"$x$")
        plt.subplot(212)
        plt.plot(t, ballY, 'b1')
        plt.plot(t, modelY, 'k')
        plt.xlabel(r"$t$ (s)")
        plt.ylabel(r"$y$")
        plt.savefig('/Users/sammay/Desktop/SPINLab/DigiRo/'+fileName+'_cartesian.pdf', format = 'pdf', dpi =1200)   
    
    cv2.destroyAllWindows()
    vid.release()
    video_writer.release()

root = Tk()
root.title('DigiPyRo')
startButton = Button(root, text = "Start!", command = start)
startButton.grid(row=6, column=0)
digiRPMVar = DoubleVar()
physRPMVar = DoubleVar()
digiRPMEntry = Entry(root, textvariable=digiRPMVar)
physRPMEntry = Entry(root, textvariable=physRPMVar)
digiLabel = Label(root, text="Additional digital rotation (RPM):")
physLabel = Label(root, text="Physical rotation (RPM):")
digiRPMEntry.grid(row=1, column=1)
physRPMEntry.grid(row=0, column=1)
digiLabel.grid(row=1, column=0)
physLabel.grid(row=0, column=0)

filenameVar = StringVar()
filenameEntry = Entry(root, textvariable = filenameVar)
filenameLabel = Label(root, text="Full filepath to movie:")
filenameEntry.grid(row=2, column=1)
filenameLabel.grid(row=2, column=0)

savefileVar = StringVar()
savefileEntry = Entry(root, textvariable = savefileVar)
savefileLabel = Label(root, text="Save output video as:")
savefileEntry.grid(row=3, column=1)
savefileLabel.grid(row=3, column=0)

startTimeVar = DoubleVar()
endTimeVar = DoubleVar()
startTimeEntry = Entry(root, textvariable = startTimeVar)
endTimeEntry = Entry(root, textvariable = endTimeVar)
startTimeLabel = Label(root, text="Start and end times (in seconds):")
startTimeLabel.grid(row=4, column=0)
startTimeEntry.grid(row=4, column=1)
endTimeEntry.grid(row=4, column=2)

trackVar = BooleanVar()
trackEntry = Checkbutton(root, text="Track Ball", variable=trackVar)
trackEntry.grid(row=3, column=2)

fpsVar = DoubleVar()
fpsEntry = Entry(root, textvariable=fpsVar)
fpsLabel = Label(root, text="Frames per second of video:")
fpsEntry.grid(row=5, column=1)
fpsLabel.grid(row=5, column=0)

root.mainloop()
