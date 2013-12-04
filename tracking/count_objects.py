import os
import sys
import cv2 as cv
import numpy as np
from collections import deque

FRAMES_LIMIT = 1000

MAX_BG_WINDOW_WIDTH = 500

MASK_THRESHOLD = 20


def err(str):
    print >> sys.stderr, str


class SlidingWindowBGFinder():
    def __init__(self, videoFpath, windowWidth=MAX_BG_WINDOW_WIDTH):
        self.MAX_WIDTH = windowWidth

        self.video = cv.VideoCapture(videoFpath)
        if not self.video.isOpened():
            err('Error in SlidingWindowBGFinder: Video could not be opened')
            exit(1)

        ok, frame = self.video.read()
        if not ok:
            err('Error in SlidingWindowBGFinder: Video is empty')
            exit(1)

        self.windowFrames = deque()

        self.windowSum = np.zeros(frame.shape, np.float)

        self.windowWidth = 0
        while self.windowWidth < self.MAX_WIDTH:
            self.windowWidth += 1

            self.windowSum += frame

            self.windowFrames.append(frame)

            ok, frame = self.video.read()
            if not ok:
                break

        self.background = np.empty_like(frame)

        self.curPos = 0


    def getNextBackground(self):
        self.curPos += 1

        np.copyto(self.background, self.windowSum / self.windowWidth, casting='unsafe')

        if self.curPos > self.MAX_WIDTH / 2:
            ok, nextFrame = self.video.read()
            if ok:
                # Slide forward:
                leftFrame = self.windowFrames.popleft()
                self.windowFrames.append(nextFrame)
                self.windowSum -= leftFrame
                self.windowSum += nextFrame

        return self.background


    def getCurPos(self):
        return self.curPos


# class FixedBGFinder():
# 	def __init__(self, video_f):
# 		self.video_f = video_f

# 	def find_bg(frame):
# 		ok, frame = video_f.read()
# 		if not ok:
# 			return None

# 		accum = np.zeros(frame.shape, np.float)

# 		for k in range(1, FRAMES_LIMIT):
# 			accum += frame

# 			is_read, frame = video_f.read()
# 			if not is_read:
# 				break

# 		mean = np.empty_like(accum)
# 		np.copyto(mean, accum / k, casting='unsafe')
# 		return mean



# def evaluate_background(video_f, window_width=0):
# 	ok, frame = video_f.read()
# 	if not ok:
# 		return None

# 	accum = np.zeros(frame.shape, np.float)

# 	for k in range(1, FRAMES_LIMIT):
# 		print k

# 		accum += frame

# 		is_read, frame = video_f.read()
# 		if not is_read:
# 			break

# 	mean = np.empty_like(accum)
# 	np.copyto(mean, accum / k, casting='unsafe')
# 	return mean


def processFGMask(fg_mask, wait_q=False):
    fg_mask[fg_mask < MASK_THRESHOLD] = 0

    cv.imshow('thresholded', fg_mask)

    eroded = cv.erode(fg_mask, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))

    cv.imshow('eroded', eroded)

    eroded_dilated = cv.dilate(eroded, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))

    cv.imshow('eroded_dilated', eroded_dilated)

    return eroded


# def process_contours(contours, wait_q=False):
# 	frame_countors = fr	ame.copy()
# 	cv.drawContours(frame_countors, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)
# 	cv.imshow('contours', frame_countors)
# 	if wait_q and cv.waitKey() == 113:
# 		exit(1)

# 	frame_countors_eroded = frame.copy()
# 	contours_eroded = cv.erode(contours, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))
# 	cv.drawContours(frame_countors_eroded, contours_eroded, contourIdx=-1, color=(0, 0, 255), thickness=1)
# 	cv.imshow('contours eroded', frame_countors_eroded)
# 	if wait_q and cv.waitKey() == 113:
# 		exit(1)

# 	frame_countors_dilated = frame.copy()
# 	contours_dilated = cv.dilate(contours, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))
# 	cv.drawContours(frame_countors_dilated, contours_dilated, contourIdx=-1, color=(0, 0, 255), thickness=1)
# 	cv.imshow('contours dilated', frame_countors_dilated)
# 	if wait_q and cv.waitKey() == 113:
# 		exit(1)

# 	frame_countors_eroded_dilated = frame.copy()
# 	contours_eroded_dilated = cv.dilate(contours_eroded, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))
# 	cv.drawContours(frame_countors_eroded_dilated, contours_eroded_dilated, contourIdx=-1, color=(0, 0, 255),
# thickness=1)
# 	cv.imshow('contours eroded dilated', frame_countors_eroded_dilated)
# 	if wait_q and cv.waitKey() == 113:
# 		exit(1)


# def get_bg():
# 	if os.path.isfile(BACKGROUND_FPATH):
# 		background = cv.imread(BACKGROUND_FPATH)
# 	else:
# 		background = evaluate_background(video_f, SLIDING_WINDOW_WIDTH)

# 	if background is None:
# 		err('Video could not be read: no frames')
# 		exit(1)

# 	cv.imshow('background', background)

#  	return background


# def substruct_foreground(frame, background):
# 	bgGray = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
# 	frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# 	return cv.absdiff(frameGray, bgGray)

# diff = np.zeros(frame_gray.shape, np.int16)
# diff += frame_gray
# diff -= bg_gray
# diff = np.absolute(diff)

# fg_mask = np.zeros(frame_gray.shape, np.uint8)
# np.copyto(fg_mask, diff, casting='unsafe')
# return fg_mask


# mog = cv.BackgroundSubtractorMOG()
# def substruct_with_bgs(frame, background):
# 	if substruct_with_bgs.first_substruction:
# 		# mog.apply(background)
# 		substruct_with_bgs.first_substruction = False

# 	return mog.apply(frame)

# substruct_with_bgs.first_substruction = True


# def process_frame(frame, background, wait_q):	
# 	cv.imshow('frame', frame)

# 	fg_mask = cv.cvtColor(cv.absdiff(frame, background), cv.COLOR_BGR2GRAY) 

# 	fg_mask = process_fg_mask(fg_mask, '', wait_q)

# 	# fg_mask = substruct_with_bgs(frame, background)
# 	# fg_mask_auto = process_fg_mask(fg_mask_auto, 'auto ', wait_q)

# 	# fg_mask_eroded = cv.erode(fg_mask, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))

# 	# background = bgs.getBackgroundImage()

# 	contours, _ = cv.findContours(fg_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# 	# contours_auto, _ = cv.findContours(fg_mask_auto.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)



def main(args):
    if len(args) >= 1:
        videoFpath = args[0]
    else:
        err('Please provide video file')
        exit(1)

    video = cv.VideoCapture(videoFpath)
    if not video.isOpened():
        err('Video could not be read')
        exit(1)

    bgFinder = SlidingWindowBGFinder(videoFpath)

    for i in range(1, FRAMES_LIMIT):
        print i

        ok, frame = video.read()
        if not ok:
            print 'Video ends'
            break
        cv.imshow('frame', frame)

        background = bgFinder.getNextBackground()
        cv.imshow('background', background)

        diff = cv.absdiff(frame, background)
        fgMask = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        cv.imshow('fgMask', fgMask)

        fgMask = processFGMask(fgMask)
        _, fgMask = cv.threshold(fgMask, MASK_THRESHOLD, 255, cv.THRESH_BINARY)
        cv.imshow('binary_mask', fgMask)

        contours, _ = cv.findContours(fgMask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(frame, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)
        cv.imshow('contours', frame)

        # h, w, c = frame.shape
        # floodFillMask = np.zeros((h + 2, w + 2, c), np.uint8)
        # cv.floodFill(fgMask, floodFillMask, (0, 0), (255, 255, 255))

        if cv.waitKey() in (ord('q'), 27):
            break

    cv.imwrite('fgMask.bmp', fgMask)


if __name__ == '__main__':
    main(sys.argv[1:])

