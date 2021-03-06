#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from collections import defaultdict
from PIL import Image

roi_defined = False

def define_ROI(event, x, y, flags, param):
    '''
    Define the region of intetest
    '''
    global r, c, w, h, roi_defined
    # if the left mouse button was clicked,
    # record the starting ROI coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2 - r)
        w = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)
        roi_defined = True

def gradient_orientation(img):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    img1 = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    img2 = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    orientation = np.arctan2(img2, img1)
    return orientation

def gradient_norme(img):
    '''
    Calculate the gradient norme for edge point in the image
    '''
    img1 = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    img2 = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    norme = np.hypot(img1, img2)
    return norme

def plot_orientation(frame, norme, orientation, orientation__):
    '''
    Plot the gradient orientation for edge point in the image
    '''
    fig = plt.figure(figsize=(12, 8))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.imshow(frame)
    plt.title('Original')

    plt.subplot(222)
    plt.imshow(orientation, cmap='gray', vmin=-math.pi, vmax=math.pi)
    plt.title('Orientation de gradient')

    plt.subplot(223)
    plt.imshow(norme, cmap='gray', vmin=0.0, vmax=255.0)
    plt.title('Norme de gradient')

    plt.subplot(224)
    plt.imshow(orientation_.transpose(1,2,0), cmap='gray', vmin=0, vmax=255)
    plt.title('Orientation')

    plt.show()

def get_index_hough(norme, orientation_clone, th_min):
    '''
    Get the index for hough vote (plot)
    '''
    index = np.where(norme < th_min)
    orientation_clone[0][index] = 245  # red
    orientation_clone[1][index] = 0
    orientation_clone[2][index] = 0
    return orientation_clone, index

def get_index_hough2(norme, th_min):
    '''
    Get the index for hough vote (by changing the value to 0)
    '''
    norme_hough =  norme.copy()
    norme_hough[norme_hough[:, :] < th_min] = 0
    return norme_hough

def build_r_table(img):
    '''
    Build the R-table from the given shape image and a reference point
    '''
    img_center = [int(img.shape[0] / 2), int(img.shape[1] / 2)]
    img = np.uint8(img)

    edges = img

    gradient = gradient_orientation(edges) * 180 // np.pi

    r_table = defaultdict(list)
    for (i, j), value in np.ndenumerate(edges): #edge
        if value:
            r_table[gradient[i, j]].append((img_center[0] - i, img_center[1] - j))
    return r_table

def matchTable(img, table):
    """
    Accumulator with searched votes
    """
    # matches the reference table with the given input
    # image for testing generalized Hough Transform
    acc = np.zeros((img.shape[0]+int(0.2*img.shape[0]), img.shape[1]+int(0.2*img.shape[0])))  # acc array requires some extra space

    gradient = gradient_orientation(img) * 180 // np.pi

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0:  # boundary point
                theta = gradient[i, j]
                vectors = table[theta]
                for vector in vectors:
                    acc[vector[0]+i, vector[1]+j] += 1
    return acc

def findMax(acc):
    """
    Find the index of the maximum value
    """
    ridx, cidx = np.unravel_index(acc.argmax(), acc.shape)
    return [acc[ridx, cidx], ridx, cidx]


cap = cv2.VideoCapture('./Sequences/Antoine_Mug.mp4')  #Antoine_Mug ./Sequences/VOT-Ball.mp4
# cap = cv2.VideoCapture(0)

th_min = 60 # threshod min
# take first frame of the video
ret, frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the ROI is defined, draw it!
    if (roi_defined):
        # draw a green rectangle around the region of interest
        cv2.rectangle(frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
    # else reset the image...
    else:
        frame = clone.copy()
    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

track_window = (r, c, h, w)

# # set up the ROI for tracking
clone = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
orientation = gradient_orientation(clone)
norme = gradient_norme(clone)

orientation_clone = np.asarray([orientation.copy(), orientation.copy(), orientation.copy()])
orientation_, index_hough = get_index_hough(norme, orientation_clone, th_min)

norme = get_index_hough2(norme, th_min)

roi = norme[c:c + w, r:r + h]
Rtable = build_r_table(roi)
acc = matchTable(norme, Rtable)

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

## General Hough vote
maxval, ridx, cidx = findMax(acc)
track_window = ridx - h//2, cidx - w//2, h, w

# Meanshift
# ret, track_window = cv2.meanShift(acc, track_window, term_crit)

# plot_orientation(frame, norme, orientation, orientation_)

cpt = 1
while (1):
    ret, frame = cap.read()
    if ret == True:
        clone = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        orientation = gradient_orientation(clone)
        norme = gradient_norme(clone)
        orientation_clone = np.asarray([orientation.copy(), orientation.copy(), orientation.copy()])
        orientation_, index_hough = get_index_hough(norme, orientation_clone, th_min)
        norme = get_index_hough2(norme, th_min)

        ## Update of ROI
        # roi = norme[c:c + w, r:r + h]
        # Rtable = build_r_table(roi)
        acc = matchTable(norme, Rtable)


        ## General Hough vote
        maxval, ridx, cidx = findMax(acc)
        track_window = ridx - h//2, cidx - w//2, h, w

        ## Meanshift
        # ret, track_window = cv2.meanShift(acc, track_window, term_crit)

        r, c, h, w=  track_window
        frame_tracked = cv2.rectangle(frame, (r, c), (r + h, c + w), (255, 0, 0), 2)
        cv2.imshow('Sequence', frame_tracked)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            # cv2.imwrite('R_H_%04d.png' % cpt, dst)
            cv2.imwrite('Frame_%04d.png' % cpt, frame_tracked)
            cv2.imshow("Vote de Hough", acc)
            plot_orientation(frame, norme, orientation, orientation_)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()