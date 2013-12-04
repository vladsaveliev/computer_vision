import sys
import cv2 as cv
import numpy as np


def filter_matches_by_min_dist(matches, k=3):
    dists = [m.distance for m in matches]
    return [m for m in matches if m.distance < k * min(dists)]


def filter_matches_by_avg_dist(matches, k=0.15):
    dists = [m.distance for m in matches]
    threshold = (sum(dists) / len(dists)) * k
    return [m for m in matches if m.distance < threshold]


def glue(img1, img2):
    detector = cv.SURF(1500)
    kps1, des1 = detector.detectAndCompute(img1, None)
    kps2, des2 = detector.detectAndCompute(img2, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = filter_matches_by_min_dist(matcher.match(des1, des2))

    pts1 = np.float32([kps1[int(m.queryIdx)].pt for m in matches])
    pts2 = np.float32([kps2[int(m.trainIdx)].pt for m in matches])

    H, _ = cv.findHomography(pts2, pts1, cv.RANSAC)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    result = cv.warpPerspective(src=img2, M=H, dsize=(w1 + w2, max(h1, h2)))
    result[:h1, :w1] = img1
    return result


def main(args):
    args = args or ['hw5_panorams_image1.jpg', 'hw5_panorams_image2.jpg']

    cv.imwrite('hw5_panoram_left_to_right.jpg', reduce(glue, map(cv.imread, args)))

    images = map(cv.imread, args)
    n = len(images)
    left_half = reduce(lambda r, l: glue(l, r), images[:n/2][::-1])
    panorama = reduce(glue, [left_half] + images[n/2:])
    cv.imwrite('hw5_panoram_from_center.jpg', panorama)

    cv.imshow('panoram', panorama)
    cv.waitKey()


if __name__ == '__main__':
    main(sys.argv[1:])


