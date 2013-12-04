import math
from numpy.matlib import rand
import os
import sys
import cv2 as cv
import numpy as np


def main(args):
    #circles_img = np.zeros((500, 500), np.uint8)
    #cv.circle(circles_img, (250, 250), 50, 255)
    #cv.imwrite('circles.bmp', circles_img)
    #
    #circles = cv.HoughCircles(circles_img, method=3, dp=1, minDist=20, param1=50, param2=30)
    #print circles

    h, w = 150, 150
    lines_img = np.zeros((h, w), np.uint8)
    cv.line(lines_img, pt1=(65, 189), pt2=(10, 89), color=255, thickness=1)
    #cv.line(lines_img, pt1=(100, 250), pt2=(450, 250), color=255, thickness=3)
    cv.imwrite('lines.bmp', lines_img)

    #lines = cv.HoughLines(lines_img, rho=450, theta=200, threshold=1)
    #print lines

    lines = hough_lines(lines_img)
    print '\n'.join(map(str, lines))
    for rho, theta in lines:
        cth = math.cos(theta)
        sth = math.sin(theta)
        x, y = rho * cth, rho * sth
        x1, y1 = x + w * (-sth), y + h * cth
        x2, y2 = x - w * (-sth), y - h * cth
        pt1 = int(x1), int(y1)
        pt2 = int(x2), int(y2)
        cv.line(lines_img, pt1, pt2, color=180, thickness=1)

    cv.imwrite('lines_found.bmp', lines_img)


def hough_lines(img, threshold=25):
    h, w = img.shape
    max_rho = int(math.sqrt(h * h + w * w))
    rhos_indexes = dict([(rho, i) for i, rho in enumerate(np.arange(-max_rho, max_rho, step=1))])
    print rhos_indexes
    print len(rhos_indexes)
    thetas = np.linspace(-math.pi, math.pi, num=200)
    print len(thetas)
    accum = np.zeros((len(rhos_indexes), len(thetas)), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            if img[y, x] > 0:
                #print x, y
                for th_i in range(len(thetas)):
                    #print 'th_i=', th_i,
                    theta = thetas[th_i]
                    #print 'theta=', theta,
                    rho = x * math.cos(theta) + y * math.sin(theta)
                    #print 'rho=', rho,
                    rh_i = rhos_indexes[int(rho)]
                    #print 'rh_i=', rh_i,
                    accum[rh_i, th_i] = min(int(accum[rh_i, th_i]) + 1, 255)
                #print 'accum[rh_i, th_i]=', accum[rh_i, th_i]
                #print ''

    cv.imwrite('accum.bmp', accum)
    _, accum_mask = cv.threshold(accum, thresh=0, maxval=255, type=cv.THRESH_BINARY)
    cv.imwrite('accum_mask.bmp', accum)

    lines = []
    for rho, rh_i in rhos_indexes.items():
        for th_i, theta in enumerate(thetas):
            if accum[rh_i, th_i] > threshold:
                lines.append((rho, theta))

    return lines


if __name__ == '__main__':
    main(sys.argv[1:])
