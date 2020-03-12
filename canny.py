# Michael Harris
# CAP 4453 - Robot Vision - Spring 2019
# PA1 - Canny Edge Detection

# RESULTS: the effect of the sigma chosen on the image when performing canny edge detection is
# important, if the image is too noisy then false edges will be picked up and edge detection will
# not be reliable. for noisy images, a sigma of two or three is ideal. for these images, I used
# a sigma of one since there was very little noise and a light smoothing was all that was needed
# for reliable detection of edges. with a sigma of 2, the gradient images felt a little too blurry
# and there was no clear advantage. this could be due to some inherent tolerance for noise in the
# canny algorithm, by using derivatives, localization and double thresholding to remove streaking.


import numpy as np
from scipy.ndimage.filters import convolve, convolve1d
from scipy.misc import imread, imshow
from matplotlib import pyplot as plot
from math import ceil, sqrt, pi

# return a 1-dimensional array representation of a gaussian distribution, based on sigma
def Gaussian1(o):
    # rule of thumb is a window size of 6 sigma...
    len = int(ceil(float(o) * 6))

    # .. +1
    if (len % 2 == 0):
        len += 1

    # find the index location of 0
    cen = int(len/2)

    # need 2*cen elements, +1 to account for 0
    x = np.arange(-cen, cen + 1)

    # here's where the 1D Gaussian magic happens, store in result
    result = np.exp(-(x**2)/(2*o**2))

    # normalize so total === 1
    result /= np.sum(result)
    return result



# open images
im_1 = imread("canny1.jpg", mode="L")
im_1 = np.array(im_1, dtype=float)
im_2 = imread("canny2.jpg", mode="L")
im_2 = np.array(im_2, dtype=float)

# prepare images to show gradient magnitude
im_gm_1 = im_1.copy() #gradient magnitude using central differences
im_gm_2 = im_2.copy() #gradient magnitude using central differences

im_1_orig = im_1.copy()
im_2_orig = im_2.copy()

# create 1d gaussian kernel with sigma of 1
k = Gaussian1(1)

# do each 1 dimensional gaussian filter
convolve1d(im_1, k, -1, im_1)
convolve1d(im_1, k, 0, im_1)
convolve1d(im_2, k, -1, im_2)
convolve1d(im_2, k, 0, im_2)

# sobel filtering of blurred images to obtain gradient images
im_1_x = convolve(im_1, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
im_1_y = convolve(im_1, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
im_2_x = convolve(im_2, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
im_2_y = convolve(im_2, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# gradient magnitudes
for y in range(0,im_1.shape[0]):
    for x in range(0,im_1.shape[1]):
        im_gm_1[y][x] = sqrt(pow(im_1_x[y][x],2) + pow(im_1_y[y][x],2))

for y in range(0,im_2.shape[0]):
    for x in range(0,im_2.shape[1]):
        im_gm_2[y][x] = sqrt(pow(im_2_x[y][x],2) + pow(im_2_y[y][x],2))

# gradient orientation
o1 = np.arctan2(im_1_y, im_1_x)
o2 = np.arctan2(im_2_y, im_2_x)

# quantize gradient orientation for suppression
o1_d = (np.round(o1 * (5.0 / pi)) + 5) % 5
o2_d = (np.round(o2 * (5.0 / pi)) + 5) % 5

# suppression
im_1_sup = im_gm_1.copy()
im_2_sup = im_gm_2.copy()

# loop through image, mark edges as suppressed. if gradient responses
# at r and p are smaller than q, q is an edge
for y in range(im_1.shape[0]):
	for x in range(im_1.shape[1]):
		# edge, mark as non-max and skip loop iteration
		if y == 0 or (y == im_1.shape[0] - 1) or x == 0 or (x == im_1.shape[1] - 1):
			im_1_sup[y][x] = 0
			continue

		# get direction and suppress if needed
		direction = o1_d[y][x] % 4
		if direction == 0: # horizontal (x-dir)
			if im_gm_1[y][x] <= im_gm_1[y][x-1] or im_gm_1[y][x] <= im_gm_1[y][x+1]:
				im_1_sup[y][x] = 0
		if direction == 1: # positive slope
			if im_gm_1[y][x] <= im_gm_1[y-1][x+1] or im_gm_1[y][x] <= im_gm_1[y+1][x-1]:
				im_1_sup[y][x] = 0
		if direction == 2: # vertical (y-dir)
			if im_gm_1[y][x] <= im_gm_1[y-1][x] or im_gm_1[y][x] <= im_gm_1[y+1][x]:
				im_1_sup[y][x] = 0
		if direction == 3: # negative slope
			if im_gm_1[y][x] <= im_gm_1[y-1][x-1] or im_gm_1[y][x] <= im_gm_1[y+1][x+1]:
				im_1_sup[y][x] = 0

for y in range(im_2.shape[0]):
	for x in range(im_2.shape[1]):
		# edge, mark as non-max and skip loop iteration
		if y == 0 or (y == im_2.shape[0] - 1) or x == 0 or (x == im_2.shape[1] - 1):
			im_2_sup[y][x] = 0
			continue

		direction = o2_d[y][x] % 4
		if direction == 0: # horizontal (x-dir)
			if im_gm_2[y][x] <= im_gm_2[y][x-1] or im_gm_2[y][x] <= im_gm_2[y][x+1]:
				im_2_sup[y][x] = 0
		if direction == 1: # positive slope
			if im_gm_2[y][x] <= im_gm_2[y-1][x+1] or im_gm_2[y][x] <= im_gm_2[y+1][x-1]:
				im_2_sup[y][x] = 0
		if direction == 2: # vertical (y-dir)
			if im_gm_2[y][x] <= im_gm_2[y-1][x] or im_gm_2[y][x] <= im_gm_2[y+1][x]:
				im_2_sup[y][x] = 0
		if direction == 3: # negative slope
			if im_gm_2[y][x] <= im_gm_2[y-1][x-1] or im_gm_2[y][x] <= im_gm_2[y+1][x+1]:
				im_2_sup[y][x] = 0

# thresholds, cut off below these. using a value of 2x instead of 1.5 as reccomended in the slides
threshold1_upper = im_1_sup > 80
threshold1_lower = im_1_sup > 40

threshold2_upper = im_2_sup > 80
threshold2_lower = im_2_sup > 40

# 2 tiers of edges within thresholds
weights1 = np.array(threshold1_upper, dtype=np.uint8) + threshold1_lower
weights2 = np.array(threshold2_upper, dtype=np.uint8) + threshold2_lower

# hystersis, remove non-edge points that are not connected to a strong edge
edges = threshold1_upper.copy()
pixels = []
for y in range(1, im_1.shape[0]-1):
	for x in range(1, im_1.shape[1]-1):
		# not weak, skip iteration
		if weights1[y][x] != 1:
			continue

		# get window, find max, add to edges if strong pixel (===2)
		window = weights1[y-1:y+2,x-1:x+2]

		max = window.max()

		if max == 2:
			pixels.append((y, x))
			edges[y][x] = 1

# extend edges if nearby strong pixel
while len(pixels) > 0:
	pixels2 = []
	for y, x in pixels:
		for dY in range(-2, 2):
			for dX in range(-2, 2):
				# checking position with itself so skip it
				if dY == 0 and dX == 0:
					continue

				# check nearby positions, gather coords
				y_ = y+dY
				x_ = x+dX

				# weak pixel, not connected to a strong edge
				if weights1[y_, x_] == 1 and edges[y_, x_] == 0:
					# add it to the final image
					pixels2.append((y_, x_))
					edges[y_, x_] = 1

    # prepare for next loop
	pixels = pixels2



# hystersis, remove non-edge points that are not connected to a strong edge
edges2 = threshold2_upper.copy()
pixels = []
for y in range(1, im_2.shape[0]-1):
	for x in range(1, im_2.shape[1]-1):
		# not weak, skip iteration
		if weights2[y][x] != 1:
			continue

		# get window, find max, add to edges if strong pixel (===2)
		window = weights2[y-1:y+2,x-1:x+2]

		max = window.max()

		if max == 2:
			pixels.append((y, x))
			edges2[y][x] = 1

# extend edges if nearby strong pixel
while len(pixels) > 0:
	pixels2 = []
	for y, x in pixels:
		for dY in range(-2, 2):
			for dX in range(-2, 2):
				# checking position with itself so skip it
				if dY == 0 and dX == 0:
					continue

				# check nearby positions, gather coords
				y_ = y+dY
				x_ = x+dX

				# weak pixel, not connected to a strong edge
				if weights2[y_, x_] == 1 and edges2[y_, x_] == 0:
					# add it to the final image
					pixels2.append((y_, x_))
					edges2[y_, x_] = 1

    # prepare for next loop
	pixels = pixels2


# images
plot.subplot(441),plot.imshow(im_1_orig, cmap='gray'),plot.title('1')
plot.xticks([]), plot.yticks([]) #remove x and y values from each 'graph'
plot.subplot(442),plot.imshow(im_1, cmap='gray'),plot.title('1sig gaussian')
plot.xticks([]), plot.yticks([])
plot.subplot(443),plot.imshow(im_1_x, cmap='gray'),plot.title('x gradient')
plot.xticks([]), plot.yticks([])
plot.subplot(444),plot.imshow(im_1_y, cmap='gray'),plot.title('y gradient')
plot.xticks([]), plot.yticks([])
plot.subplot(446),plot.imshow(im_gm_1, cmap='gray'),plot.title('grad magnitude')
plot.xticks([]), plot.yticks([])
plot.subplot(447),plot.imshow(o1, cmap='gray'),plot.title('grad angle (arctan)')
plot.xticks([]), plot.yticks([])
plot.subplot(448),plot.imshow(edges, cmap='gray'),plot.title('canny')
plot.xticks([]), plot.yticks([]) #remove x and y values from each 'graph'
#
plot.subplot(4,4,9),plot.imshow(im_2_orig, cmap='gray'),plot.title('2')
plot.xticks([]), plot.yticks([]) #remove x and y values from each 'graph'
plot.subplot(4,4,10),plot.imshow(im_2, cmap='gray'),plot.title('3sig gaussian')
plot.xticks([]), plot.yticks([])
plot.subplot(4,4,11),plot.imshow(im_2_x, cmap='gray'),plot.title('x gradient')
plot.xticks([]), plot.yticks([])
plot.subplot(4,4,12),plot.imshow(im_2_y, cmap='gray'),plot.title('y gradient')
plot.xticks([]), plot.yticks([])
plot.subplot(4,4,14),plot.imshow(im_gm_2, cmap='gray'),plot.title('grad magnitude')
plot.xticks([]), plot.yticks([])
plot.subplot(4,4,15),plot.imshow(o2, cmap='gray'),plot.title('grad angle (arctan)')
plot.xticks([]), plot.yticks([])
plot.subplot(4,4,16),plot.imshow(edges2, cmap='gray'),plot.title('canny')
plot.xticks([]), plot.yticks([]) #remove x and y values from each 'graph'

# show
plot.show()
