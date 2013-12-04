import cv2
import numpy as np

fileName = 'Cam5_Outdoor.avi'
cv2.namedWindow("slonik")
vc = cv2.VideoCapture(fileName)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

rval, base = vc.read()

if not rval:
    exit()

base = np.float32(base)

index = 0
framesToAverage = 250
while True:
    isRead, frame = vc.read()

    if not isRead:
        break

    cv2.accumulateWeighted(frame, base, 0.1)

    index += 1

    if index > framesToAverage:
        break

res = cv2.convertScaleAbs(base)

vc = cv2.VideoCapture(fileName)

threshold = 10
while True:
    isRead, src = vc.read()

    if not isRead:
        break

    frame = src
    frame = cv2.absdiff(frame, res)


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (threshold, frame) = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)

    frame = cv2.GaussianBlur(frame, (7,7), 0)
    frame = cv2.erode(frame, None, 10)
    frame = cv2.dilate(frame, None, 10)
    frame = cv2.GaussianBlur(frame, (7,7), 0)
    cv2.floodFill(frame, None, (0, 0), 255) 

    contours, hierarchy = cv2.findContours(frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cv2.drawContours(frame, contours, i, (255, 255, 255), 10, 8, hierarchy, 0)

    frame = cv2.erode(frame, None, 10)
    frame = cv2.dilate(frame, None, 10)
    frame = cv2.GaussianBlur(frame, (7,7), 0)
    cv2.floodFill(frame, None, (0, 0), 255) 

    contours, hierarchy = cv2.findContours(frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    area_threshold = 2000

    if len(contours) > 0: 
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if w * h > area_threshold:
                cv2.rectangle(src, (x,y), (x + w, y + h), (0,0,255), 2)

            # cv2.drawContours(src, contours, i, (255, 255, 255), 10, 8, hierarchy, 0)

    cv2.imshow('slonik', src)

    key = cv2.waitKey(27)

    if key == ord('q'):
        break

