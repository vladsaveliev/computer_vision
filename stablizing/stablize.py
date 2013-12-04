import os
import sys
import cv2 as cv
import numpy as np


def findHomography(img0, img1):
    detector = cv.FastFeatureDetector(10)
    points = np.float32([keyPoint.pt for keyPoint in detector.detect(img0)])

    nextPoints, status, _ = cv.calcOpticalFlowPyrLK(img0, img1, points)
    # points = points[np.reshape(status > 0, status.size)]
    # nextPoints = nextPoints[np.reshape(status > 0, status.size)]

    H, mask = cv.findHomography(points, nextPoints, cv.RANSAC)
    return H


def main(args):
    if len(args) >= 1:
        video_fpath = args[0]
    else:
        print >> sys.stderr, 'Please provide video file'
        return 1

    video_f = cv.VideoCapture(video_fpath)

    isRead, initFrame = video_f.read()
    if not isRead:
        print >> sys.stderr, 'Video could not be read, no frames'
        return 1

    h, w = initFrame.shape[:2]

    base, ext = os.path.splitext(video_fpath)
    out_fpath = base + '.stable' + ext
    writer = cv.VideoWriter(out_fpath, cv.cv.CV_FOURCC(*'XVID'), 24.0, (w, h), True)

    writer.write(initFrame)

    i = 1
    while True:
        print >> sys.stdout, i
        isRead, frame = video_f.read()
        i += 1

        if not isRead:
            break

        H = findHomography(frame, initFrame)
        warped_frame = cv.warpPerspective(src=frame, M=H, dsize=(w, h))

        writer.write(warped_frame)


if __name__ == '__main__':
    exit(main(sys.argv[1:]))