#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2

path = './data/Market-1501-v15.09.15/bounding_box_train/'

imagelist = os.listdir(path)

# print(imagelist)

for imgname in imagelist:
    if (imgname.endswith(".jpg")):
        image = cv2.imread(path + imgname)
        cv2.imshow("Pictures", image)
        k = cv2.waitKey(100)
        # press key "esc" to stop
        if k == 27:
            break

cv2.destroyAllWindows()