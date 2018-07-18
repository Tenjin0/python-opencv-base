import cv2
import numpy as np
from scipy import ndimage
import os

kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8,	-1],
                       [-1,	-1,	-1]])
kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1,	-1,	-1,	-1,	-1]])


# Load the data...
img = cv2.imread(os.path.join(os.getcwd(), "images", "lena.png"),

# A very simple and very narrow highpass filter
kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8,	-1],
                       [-1,	-1,	-1]])

highpass_3x3 = ndimage.convolve(data, kernel)
plot(highpass_3x3, 'Simple 3x3 Highpass')

# A slightly "wider", but sill very simple highpass filter
kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
highpass_5x5 = ndimage.convolve(data, kernel)
plot(highpass_5x5, 'Simple 5x5 Highpass')

# Another way of making a highpass filter is to simply subtract a lowpass
# filtered image from the original. Here, we'll use a simple gaussian filter
# to "blur" (i.e. a lowpass filter) the original.
lowpass = ndimage.gaussian_filter(data, 3)
gauss_highpass = data - lowpass
plot(gauss_highpass, r'Gaussian Highpass, $\sigma = 3 pixels$')

plt.show()
