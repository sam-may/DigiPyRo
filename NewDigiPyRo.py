import numpy as np
from Tkinter import *
import matplotlib
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import leastsq
import time

import cv2

matplotlib.use("Agg")

########################
### Helper Functions ###
########################

### Helper Functions: Section 1 -- User-Interaction Functions ###
# The majority of functions in this section relate to user-identification of 
# the region of interest (ROI) which will be digitally rotated, 
# OR the intialization of single-particle tracking

# Allows user to manually identify center of rotation
def center_click(event, x, y, flags, param):
    global center, frame
    clone = frame.copy()                        # save the original frame
    if event == cv2.EVENT_LBUTTONDOWN:          # if user clicks 
        center = (x,y)                          # set click location as center
        cv2.circle(frame, (x,y), 4, (255,0,0), -1) # draw circle at center
        cv2.imshow('CenterClick', frame)        # show updated image
        frame = clone.copy()                    # resets to original image so 
                                                # that if the user reselects 
                                                # the center, the old circle 
                                                # will not appear

# Shifts image so that it is centered at (x_c, y_c)
def center_image(img, x_c, y_c):
    d_x = (width/2) - x_c
    d_y = (height/2) - y_c
    shift_matrix = np.float32([[1, 0, d_x], [0, 1, d_y]])
    return cv2.warpAffine(img, shift_matrix, (width, height))

# User drags mouse and releases to indicate a conversion factor between 
# pixels and units of distance
def unit_conversion(event, x, y, flags, param):
    global frame, u_start, u_end, unit_count, unit_type, unit_conv
    clone = frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        u_start = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        u_end = (x,y)
        d2 = ((u_end[0] - u_start[0])**2) + ((u_end[1] - u_start[1])**2)
        pixel_length = (d2**(0.5))/2
        unit_conv = unit_count / pixel_length
        cv2.line(frame, u_start, u_nd, (255,0,0), 1)
        cv2.imshow('Distance Calibration', frame)
        frame = clone.copy()

# User drags mouse and releases along a diameter of the particle to set an 
# approximate size and location of particle for DPR to search for
def locate(event, x, y, flags, param):
    global frame, particle_start, particle_end, particle_center, particle_radius
    clone = frame.copy()               # save the original frame
    if event == cv2.EVENT_LBUTTONDOWN: # if user clicks
        particle_start = (x,y)
    elif event == cv2.EVENT_LBUTTONUP: # if user releases click
        particle_end = (x,y)

