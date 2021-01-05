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

# def gradient_orientation(frame):
#
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.copyMakeBorder(np.float64(frame),0,0,0,0,cv2.BORDER_REPLICATE)
#
#     hx = np.array([[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]])
#     hy = np.array([[-1, -2, -1],
#                    [0, 0, 0],
#                    [1, 2, 1]])
#
#     img1 = cv2.filter2D(frame, -1, hx)
#     img2 = cv2.filter2D(frame, -1, hy)
#
#     norme = np.sqrt(img1 * img1 + img2 * img2)
#     orientation = np.arctan2(img2, img1)
#
#     return norme, orientation

def gradient_orientation(img):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.copyMakeBorder(np.float64(img),0,0,0,0,cv2.BORDER_REPLICATE)
    img1 = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    img2 = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    orientation = np.arctan2(img2, img1)
    return orientation

def gradient_norme(img):
    '''
    Calculate the gradient norme for edge point in the image
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.copyMakeBorder(np.float64(img),0,0,0,0,cv2.BORDER_REPLICATE)
    img1 = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    img2 = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    norme = np.sqrt(img1 * img1 + img2 * img2)
    return norme

def plot_orientation(frame, norme, orientation, norme_):
    '''
    plot the gradient orientation for edge point in the image
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
    plt.imshow(norme_.transpose(1,2,0).astype('uint8'), cmap='gray', vmin=0.0, vmax=255.0)
    plt.title('Orientation')

    plt.show()

def get_small_orientation(norme, clone, threshold):
    norme_ =  np.float64([norme.copy()])
    index = np.where((norme_ < threshold))
    clone[index] = 245  # red
    clone[1][index[1:]] = 0
    clone[2][index[1:]] = 0
    return clone, index

def get_big_orientation(norme, clone, threshold):
    norme_ = np.float64([norme.copy()])
    index = np.where((norme_ > threshold))
    clone[index] = 88  # gray
    clone[1][index[1:]] = 88
    clone[2][index[1:]] = 88
    return clone, index
#
def build_r_table(img):
    '''
    Build the R-table from the given shape image and a reference point
    '''
    r_table = defaultdict(list)
    img_center = [int(img.shape[0]/2), int(img.shape[1]/2)]

    def findAngleDistance(x1, y1):
        x2, y2 = img_center[0], img_center[1]
        r = [(x2-x1), (y2-y1)]
        if (x2-x1 != 0):
            return [int(np.rad2deg(np.arctan(int((y2-y1)/(x2-x1))))), r]
        else:
            return [0, 0]

    filter_size = 3
    for x in range(img.shape[0]-(filter_size-1)):
        for y in range(img.shape[1]-(filter_size-1)):
            if (img[x, y] != 0):
                theta, r = findAngleDistance(x, y)
                if (r != 0):
                    r_table[np.absolute(theta)].append(r)

    return r_table


def matchTable(img, table, index):
    """
    param:
        im: input binary image, for searching template
        table: table for template
        index: index for vote
    output:
        accumulator with searched votes
    """
    img_center = [int(img.shape[0] / 2), int(img.shape[1] / 2)]
    acc = np.zeros((img.shape[0]+50, img.shape[1]+50))  # acc array requires some extra space
    def findGradient(x, y):
        if (x != 0):
            return int(np.rad2deg(np.arctan(int(y/x))))
        else:
            return 0
    for i in range(len(index[0])):
        m, n = (index[0][i], index[1][i])
        if img[m, n] != 0:  # boundary point
            theta = findGradient(m-img_center[0], n-img_center[1])
            vectors = table[theta]
            for vector in vectors:
                acc[vector[0]+m, vector[1]+n] += 1
    return acc

def findMax(acc):
    ridx, cidx = np.unravel_index(acc.argmax(), acc.shape)
    return [acc[ridx, cidx], ridx, cidx]

cap = cv2.VideoCapture('./Sequences/VOT-Ball.mp4')  #Antoine_Mug
# cap = cv2.VideoCapture(0)

th_min = 60 # threshod min
th_max = 200 # threshod max

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
# roi = frame[c:c + w, r:r + h]
orientation = gradient_orientation(frame)
norme = gradient_norme(frame)

clone = np.float64([norme.copy(), norme.copy(), norme.copy()])
norme_small, index_small = get_small_orientation(norme, clone, th_min)
norme_big_small, index_big = get_big_orientation(norme, norme_small, th_max)

roi = norme[c:c + w, r:r + h]
Rtable = build_r_table(roi)
acc = matchTable(norme, Rtable, index_small[1:])

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

## General Hough vote
# maxval, ridx, cidx = findMax(acc)
# track_window = ridx - h//2, cidx - w//2, h, w

# Meanshift
ret, track_window = cv2.meanShift(acc, track_window, term_crit)


cpt = 1
while (1):
    ret, frame = cap.read()
    if ret == True:

        roi = norme[c:c + w, r:r + h]
        Rtable = build_r_table(roi)
        acc = matchTable(norme, Rtable, index_small[1:])
        ## General Hough vote
        # maxval, ridx, cidx = findMax(acc)
        # track_window = ridx - h//2, cidx - w//2, h, w

        # Meanshift
        ret, track_window = cv2.meanShift(acc, track_window, term_crit)

        r, c, h, w=  track_window
        frame_tracked = cv2.rectangle(frame, (r, c), (r + h, c + w), (255, 0, 0), 2)
        cv2.imshow('Sequence', frame_tracked)

        orientation = gradient_orientation(frame)
        norme = gradient_norme(frame)
        clone = np.float64([norme.copy(), norme.copy(), norme.copy()])
        norme_small, index_small = get_small_orientation(norme, clone, th_min)
        # norme_big_small, index_big = get_big_orientation(norme, norme_small, th_max)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            # cv2.imwrite('R_H_%04d.png' % cpt, dst)
            # cv2.imwrite('Frame_%04d.png' % cpt, frame)
            plot_orientation(frame, norme, orientation, norme_big_small)
        cpt += 1
    else:
        break


cv2.destroyAllWindows()
cap.release()