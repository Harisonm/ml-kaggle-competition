import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

imgLandscape = mpimg.imread("43160967590_e901913d02_z.jpg")
# imgLandscape = mpimg.imread("test.png")

imgEdgeDetect = np.zeros((imgLandscape.shape[0], imgLandscape.shape[1], imgLandscape.shape[2]))

kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],

])
print(imgLandscape.shape)

width = imgLandscape.shape[0]
height = imgLandscape.shape[1]

for x in range(width):
    for y in range(height):
        for k in range(3):
            sumKernel = 0.0
            for i in range(3):
                for j in range(3):
                    if ((x - 1) + i in range(0, width)) & ((y - 1) + j in range(0, height)):
                        sumKernel += kernel[i][j] * imgLandscape[(x - 1) + i][(y - 1) + j][k]
            imgEdgeDetect[x][y][k] = sumKernel

mpimg.imsave(fname='imageDetectionEdge.png', arr=imgEdgeDetect, format='png')
