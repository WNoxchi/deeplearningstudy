# 2017-Oct-27 00:08
# WNixalo

# Testing out image-resize speedups without/with pillow-simd
# Image is a (1680,1050) jpg

from os.path import expanduser; import scipy.misc
import matplotlib.pyplot as plt; import cv2; import PIL
from time import time

root = expanduser('~/')
path = root + '/Deshar/DoomPy/images/Hiigaran_Hull_Paint.jpg'

averages = [0. for i in range(5)]

iters = 50
for i in range(iters):
    times = []
    t0,t1 = 0.,0.

    # Openning Image w/ PIL

    t0 = time()
    img = PIL.Image.open(path)
    t1 = time()

    imgarr = [img for i in range(50)]
    len(imgarr)
    del imgarr

    times.append(t1-t0)
    t0,t1 = 0.,0.

    # Openning Image w/ MatplotLib

    t0 = time()
    plt.imshow(img)
    t1 = time()

    times.append(t1-t0)
    t0,t1 = 0.,0.
    # plt.show()

    # Resizing Image w/ PIL
    t0 = time()
    img = img.resize((3360, 2100))
    t1 = time()

    times.append(t1-t0)
    t0,t1 = 0.,0.

    # checking it resized correctly
    # plt.imshow(img); plt.show()

    # Openning Image w/ OpenCV 3.3.0

    t0 = time()
    img = cv2.imread(path, 0)
    # cv2.imshow('', img)
    t1 = time()

    times.append(t1-t0)
    t0,t1 = 0.,0.

    # Resizing Image w/ OpenCV
    t0 = time()
    img = cv2.resize(img, (3360, 2100)) # interpolation=CV_INTER_LINEAR
    t1 = time()

    times.append(t1-t0)

    for i in range(len(times)):
        averages[i] += times[i]

for i in range(len(averages)):
    averages[i] /= iters

# just checking it resized correctly
# cv2.imshow('',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("{:<0} {:>13} {:>24} {:>16}".format('Times (Open):','PIL', 'MatplotLib', 'OpenCV'))
print("{:>36} {:>14} {:>14}".format(averages[0], averages[1], averages[3]))

print("{:<0} {:>8} {:>20}".format('Times (Resize 2x):', 'PIL', 'OpenCV'))
print("{:>36} {:>14}".format(averages[2], averages[4]))

print("Iterations: {}".format(iters))


################################################################################
# OUTPUT w/ regular PIL:
# (FAI) Waynes-MBP:Kaukasos WayNoxchi$ python pillow-simd-benchmark.py
# Times (Open):           PIL               MatplotLib           OpenCV
#                0.0011928749084472656 0.09201124668121338 0.017856025695800783
# Times (Resize 2x):      PIL               OpenCV
#                 0.013700852394104004 0.004898147583007812
# Iterations: 50
################################################################################
# OUTPUT w/ pillow-simd:
# (FAI) Waynes-MBP:Kaukasos WayNoxchi$ python pillow-simd-benchmark.py
# Times (Open):           PIL               MatplotLib           OpenCV
#                0.0012062406539916993 0.08796523094177246 0.017541275024414063
# Times (Resize 2x):      PIL               OpenCV
#                 0.010742950439453124 0.0048766183853149415
# Iterations: 50


# NOTE: I wasn't sure which operations would be sped up w/ SIMD. Realized later
#       on it was resizing. This is reflected above. The time to upscale a
#       1680x1050 image by 2X for PIL decreased by ~130μs or a ~21.6% speedup.
#       I assume this would be much more dramatic for batches of images and
#       more complicated resampling algorithms.
#       If I ran this again I'd focus only on capturing time for PIL to resize
#       batches of images. Say, from an array of 10k images or so.
# NOTE: All other times remained over multiple runs. PIL resize had a pre-SIMD
#       high of 0.019.. seconds, and a post-SIMD high of 0.01100658.. seconds.
