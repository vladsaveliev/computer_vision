import sys
import cv2 as cv
import numpy as np
import scipy as sp


def concat_images(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    comb = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    comb[:h1, :w1, :] = img1
    comb[:h2, w1:, :] = img2
    return comb


def filter_matches_by_min_dist(matches, k=3):
    dists = [m.distance for m in matches]
    return [m for m in matches if m.distance < k * min(dists)]


def filter_matches_by_avg_dist(matches, k=0.15):
    dists = [m.distance for m in matches]
    threshold = (sum(dists) / len(dists)) * k
    return [m for m in matches if m.distance < threshold]


def main(args):
    img1 = cv.imread(args[0] if args else 'hw3_sift_cameraman.bmp')
    h, w = img1.shape[:2]

    # ROTATE
    rot_m = cv.getRotationMatrix2D((h / 2, w / 2), 30, 1.0)
    img2 = cv.warpAffine(img1, rot_m, (h, w))

    # FIND KEYPOINTS
    sift = cv.SIFT()
    kps1, des1 = sift.detectAndCompute(img1, None)
    kps2, des2 = sift.detectAndCompute(img2, None)

    # MATCH
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = filter_matches_by_avg_dist(matcher.match(des1, des2))

    # DRAW MATCHES
    comb_img = concat_images(img1, img2)

    for m in matches:
        color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        kp1 = kps1[int(m.queryIdx)]
        kp2 = kps2[int(m.trainIdx)]
        pt1 = tuple(map(int, kp1.pt))
        pt2 = int(kp2.pt[0] + w), int(kp2.pt[1])
        cv.line(comb_img, pt1, pt2, color)
        cv.circle(comb_img, pt1, 2, color)
        cv.circle(comb_img, pt2, 2, color)

    cv.imshow('matches', comb_img)
    cv.imwrite('hw3_sift_matches.jpg', comb_img)
    cv.waitKey()


if __name__ == '__main__':
    main(sys.argv[1:])
