import os
import sys
import cv2 as cv
import numpy as np


class Filter():
	def __init__(self, name, method, *args, **kwargs):
		self.name = name
		self.method = method
		self.args = args
		self.kwargs = kwargs

	def apply(self, img):
		return self.method(img, *self.args, **self.kwargs)


def save_result(orig, result, filter_name, dirname=''):
	diff = cv.absdiff(orig, result)
	_, mask = cv.threshold(diff, 5, 255, cv.THRESH_BINARY)

	if dirname and not os.path.exists(dirname):
		#if os.path.exists(dirname):
		#	shutil.rmtree(dirname)
		os.makedirs(dirname)

	cv.imwrite(os.path.join(dirname, filter_name + '.bmp'), result)
	cv.imwrite(os.path.join(dirname, filter_name + '_diff.bmp'), diff)
	cv.imwrite(os.path.join(dirname, filter_name + '_diff_binary.bmp'), mask)


def main(args):
	orig = cv.imread('cameraman.bmp', cv.CV_LOAD_IMAGE_GRAYSCALE)

	filters = [
		Filter('1. box_filter_2-2', cv.boxFilter, ddepth=-1, ksize=(2, 2)),
		Filter('2. gaussian_blur_3-3_0_0', cv.GaussianBlur, ksize=(3, 3), sigmaX=0, sigmaY=0),
		Filter('3. median_blur_3', cv.medianBlur, ksize=3),
		Filter('4. bilateral_10_10', cv.bilateralFilter, d=0, sigmaColor=10, sigmaSpace=0),
		Filter('5. nl_mean', cv.fastNlMeansDenoising)]

	to_orig = [f.apply(orig) for f in filters]
	[save_result(orig, res, f.name, 'to_original') for f, res in zip(filters, to_orig)]

	noised = orig + gaussian_noise(orig, 5)
	save_result(orig, noised, 'noised')

	to_noised = [f.apply(noised) for f in filters]
	[save_result(orig, res, f.name, 'to_noised') for f, res in zip(filters, to_noised)]


def gaussian_noise(img, var):
	noise = np.empty_like(img)
	cv.randn(noise, 0, var)
	return noise


def saltpepper_noise(img):
	if len(img.shape) == 3:
		h, w, c = img.shape
	else:
		h, w = img.shape

	noise = np.random.randint(0, 255, (h, w))
	black = noise < 5
	white = noise > 250
	saltpeppered_img = img.copy()

	if len(img.shape) == 3:
		saltpeppered_img[:, :][white] = 255, 255, 255
		saltpeppered_img[:, :][black] = 0, 0, 0
	else:
		saltpeppered_img[white] = 255
		saltpeppered_img[black] = 0

	return saltpeppered_img


if __name__ == '__main__':
	main(sys.argv[1:])
