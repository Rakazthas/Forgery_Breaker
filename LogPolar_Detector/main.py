# Log-Polar detection script
# Base imports
import cv2.cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
from math import copysign,log10

# Log-Polar operations
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

blockSize = 64
logRadius = blockSize/2

imgName = input("Entrer le chemin de l'image : ")
img = cv2.imread(imgName)
h,w = img.shape[:2]


print("Largeur : {} px,hauteur:{} px".format(w,h))
px_value =img[20,100]
print("Valeur du pixel en (20,100): {}".format(px_value))
cv2.imshow('img', img)

#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()

blockPerLine = int(w/blockSize)
blocks = []

for i in range (0, h - blockSize, blockSize):
    for j in range (0, w - blockSize, blockSize):
        array = img[i:i+blockSize, j:j+blockSize]
        blocks.append(array)

cv2.imshow("block i", blocks[5 + 5*blockPerLine])
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()

blocksLog = []

for x in blocks:
    imgFl = x.astype(np.float32)
    value = np.sqrt(((imgFl.shape[0]/2.0)**2.0 + (imgFl.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(imgFl, (imgFl.shape[0]/2, imgFl.shape[1]/2), blockSize/2, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    blocksLog.append(polar_image)

cv2.imshow("block log", blocksLog[5 + 5*blockPerLine])
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

