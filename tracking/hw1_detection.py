# coding=utf-8
import os
import sys
import cv2 as cv
import numpy as np
from collections import deque
from colorsys import hsv_to_rgb

FRAMES_LIMIT = 1000
MASK_THRESHOLD = 40


def abort(msg):
    print >> sys.stderr, 'Error:', msg
    exit(1)


def generate_color(id):
    base_hue = (1.0 / 8) * (id % 8)
    hue = base_hue + (id / 8)
    hsv = hue, 1.0, 1.0
    rgb = hsv_to_rgb(*hsv)
    rgb = [int(c * 256) for c in rgb]
    return rgb


class SlidingWindowBGFinder():
    max_window = 40  # 0 for unlimited

    def __init__(self, video_fpath, max_window=None):
        if max_window is not None:
            self.max_window = max_window

        self.video = cv.VideoCapture(video_fpath)
        if not self.video.isOpened():
            abort('video can not be opened')
        ok, frame = self.video.read()
        if not ok:
            abort('video is empty')

        self.window = deque()
        self.window_sum = np.zeros(frame.shape, np.float)
        self.window_width = 0

        # Prepare window
        while self.max_window > 0 and self.window_width < self.max_window:
            self.window.append(frame)
            self.window_sum += frame
            self.window_width += 1

            ok, frame = self.video.read()
            if not ok:
                break

        self.cur_pos = 0
        self.background = np.empty_like(frame)
        np.copyto(self.background, self.window_sum / self.window_width, casting='unsafe')

    def get_next_background(self):
        self.cur_pos += 1

        if self.max_window > 0 and self.cur_pos > self.max_window / 2:
            ok, next_frame = self.video.read()
            if ok:  # Slide forward:
                left_frame = self.window.popleft()
                self.window.append(next_frame)
                self.window_sum -= left_frame
                self.window_sum += next_frame

            np.copyto(self.background, self.window_sum / self.window_width, casting='unsafe')

        return self.background

    def get_current_pos(self):
        return self.cur_pos


def process_mask(fg_mask, wait_q=False):
    fg_mask[fg_mask < MASK_THRESHOLD] = 0
    # cv.imshow('thresholded', fg_mask)

    #eroded = cv.erode(fg_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
    # cv.imshow('eroded', eroded)

    #eroded_dilated = cv.dilate(eroded, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    # cv.imshow('eroded_dilated', eroded_dilated)

    dilated = cv.dilate(fg_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4)))

    return dilated


class Area():
    def __init__(self, id, seed=None):
        self.id = id
        self.seed = seed

        self.points = []
        self.center = None

    def find_center(self):
        assert self.points

        pt_num = len(self.points)
        center_x = sum([x for x, y in self.points]) / pt_num
        center_y = sum([y for x, y in self.points]) / pt_num
        self.center = center_x, center_y


def show_areas(areas, mask, image=None):
    h, w = mask.shape
    colored_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)

    for area in areas:
        color = [int(c) for c in generate_color(area.id)]
        cv.floodFill(colored_mask, flood_fill_mask, area.seed, color)
        cv.circle(colored_mask, tuple(area.center), 2, color, -1)
        if image is not None:
            cv.circle(image, tuple(area.center), 2, color, -1)

    if image is not None:
        cv.imshow('frame', image)
    cv.imshow('colored_mask', colored_mask)


AREA_MIN_SIZE = 200


def find_areas(mask):
    areas = []

    h, w = mask.shape
    #mask = np.copy(mask)
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)

    area_id = 0
    for y in range(1, h):
        for x in range(1, w):
            if mask[y][x] == 255:
                area = Area(area_id, (x, y))
                areas.append(area)
                cv.floodFill(mask, flood_fill_mask, area.seed, area.id + 1)
                area_id += 1

    for y in range(1, h):
        for x in range(1, w):
            color = mask[y][x]
            if color != 0:
                area = areas[color - 1]
                area.points.append((x, y))

    areas = filter(lambda ar: len(ar.points) > AREA_MIN_SIZE, areas)
    areas.sort(key=lambda ar: len(ar.points), reverse=True)

    for i, area in enumerate(areas):
        area.id = i
        area.find_center()

    return areas


from munkres import Munkres
m = Munkres()
MAX_VALUE = 4294967295


def match_areas(prev_areas, next_areas):
    size = max(len(prev_areas), len(next_areas))
    matrix = np.empty((size, size), dtype=np.uint32)
    matrix.fill(MAX_VALUE)

    for i, prev_area in enumerate(prev_areas):
        for j, area in enumerate(next_areas):
            dist2 = int((area.center[0] - prev_area.center[0]) ** 2 +
                        (area.center[1] - prev_area.center[1]) ** 2) or 1
            matrix[i][j] = dist2
    print matrix

    matches = m.compute(matrix)
    print 'Matches:', matches, '\n'
    for row, column in matches:
        if column < len(next_areas):
            if row < len(prev_areas):
                next_areas[column].id = prev_areas[row].id
            else:
                next_areas[column].id = row  # generate new id


def main(args):
    if len(args) >= 1:
        video_fpath = args[0]
    else:
        assert (os.path.exists('test.avi'))
        video_fpath = 'test.avi'

    video = cv.VideoCapture(video_fpath)
    if not video.isOpened():
        abort('video could not be read')

    bg_finder = SlidingWindowBGFinder(video_fpath)

    prev_areas = None

    for i in range(1, FRAMES_LIMIT):
        print i

        ok, frame = video.read()
        if not ok:
            print 'Video ends'
            break

        background = bg_finder.get_next_background()
        # cv.imshow('background', background)

        diff = cv.absdiff(frame, background)
        gray_mask = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        gray_mask = process_mask(gray_mask)
        _, binary_mask = cv.threshold(gray_mask, MASK_THRESHOLD, 255, cv.THRESH_BINARY)

        if not prev_areas:
            areas = find_areas(binary_mask)
        else:
            areas = find_areas(binary_mask)
            match_areas(prev_areas, areas)
        prev_areas = areas

        show_areas(areas, binary_mask, frame)
        if cv.waitKey() in (ord('q'), 27) and i > 0:
            break


if __name__ == '__main__':
    main(sys.argv[1:])
























